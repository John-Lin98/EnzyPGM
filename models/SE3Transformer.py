import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_scatter import scatter_softmax

logger = logging.getLogger(__name__)


def check_nan(x):
    x = x.detach()
    try:
        return (~torch.isfinite(x)).any().item()
    except RuntimeError:
        return True


class SE3TransformerLayer(nn.Module):
    """
    简化版 SE(3)-equivariant Transformer 图层（only 标量节点特征 + 坐标）。

    目标：接口与原 E_GCL_RM_Node 尽量一致，可直接替换使用：
      - __init__ 参数不变；
      - forward(h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30)
      - 返回 (h_out, coord_out, edge_attr)

    等变性设计要点：
      1) 注意力 logit 只依赖于标量特征 q_i, k_j 和距离 r_ij（通过 radial MLP 编码），
         与坐标的具体方向无关 ⇒ 旋转/平移不改变注意力权重；
      2) 坐标更新只用 coord_diff * w(edge_feat)，其中 w 是旋转不变的标量函数，
         coord_diff 在旋转下整体乘同一个 R ⇒ 坐标更新整体等变。

    输入约定（与原 EGNN 一致）：
      - h:
          * 若是 dict: h["hidden"] 形状 [B,L,C] 或 [N,C]
          * 若是 tensor:
              - [B,L,C]：自动展平为 [B*L,C]
              - [N,C]：直接使用
      - coord: [N,3]，N = B*L
      - edge_index: (row, col)，长度 E
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=True,      # 这里始终用 attention，但保留参数兼容
        normalize=False,
        coords_agg="mean",
        tanh=False,
        cfg=None,
    ):
        super().__init__()

        if cfg is None:
            cfg = {}
        self.device = torch.device(cfg.get("device", "cpu"))

        self.input_nf = input_nf
        self.output_nf = output_nf if output_nf is not None else input_nf
        self.hidden_nf = hidden_nf if hidden_nf is not None else input_nf

        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        # ========== 1) 节点侧 Q/K/V ==========

        self.to_q = nn.Linear(self.input_nf, self.hidden_nf)
        self.to_k = nn.Linear(self.input_nf, self.hidden_nf)
        self.to_v = nn.Linear(self.input_nf, self.hidden_nf)

        # ========== 2) 距离编码：r^2 -> radial_embed -> attention bias & coord weight ==========

        # 可以换成 RBF embedding，这里先用简单 MLP
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1),  # 标量 bias
        )

        # ========== 3) 坐标更新用的标量网络：edge_feat -> w_{ij} ==========

        coord_mlp = [
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1, bias=False),  # 输出 1 维标量
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        # 小尺度初始化，避免坐标更新过猛
        last_linear = None
        for m in reversed(self.coord_mlp):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight, gain=0.001)

        # ========== 4) 节点聚合后的投影 & 门控残差 ==========

        self.node_proj = nn.Linear(self.hidden_nf, self.output_nf)

        # 若输入维度与输出维度不同，增加一个投影把 h 投到输出通道
        if self.input_nf == self.output_nf:
            self.h_in_proj = nn.Identity()
        else:
            self.h_in_proj = nn.Linear(self.input_nf, self.output_nf)

        self.node_gate = nn.Sequential(
            nn.Linear(self.output_nf, self.output_nf),
            nn.ReLU(),
            nn.Linear(self.output_nf, self.output_nf),
            nn.Sigmoid(),
        )

        # 为了和原类保持属性兼容（有些上游可能访问）
        self.attention = True  # 这个 layer 本身就是 attention
        self.use_encoder_embed = False
        self.use_entropy_feat = False
        self.node_agg = "sum"  # 或 "mean"

    # ===== geometry utils =====

    def coord2radial(self, edge_index, coord):
        """
        coord: [N,3]
        edge_index: (row, col)
        return:
          radial:    [E,1]   (r^2)
          coord_diff:[E,3]   (coord[row] - coord[col])
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]  # [E,3]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)  # [E,1]

        if self.normalize:
            norm = torch.sqrt(radial + self.epsilon)
            coord_diff = coord_diff / norm

        radial = radial.to(self.device)
        coord_diff = coord_diff.to(self.device)
        return radial, coord_diff

    # ===== attention on edges =====

    def edge_model(self, q, k, v, radial, edge_index):
        """
        q, k, v:   [N, hidden_nf]
        radial:    [E,1] (r^2)
        edge_index:(row, col)
        return:
          edge_feat: [E, hidden_nf] = α_ij * v_j
          attn:      [E] attention 权重（可选返回调试）
        """
        row, col = edge_index
        q_i = q[row]        # [E, hidden_nf]
        k_j = k[col]        # [E, hidden_nf]
        v_j = v[col]        # [E, hidden_nf]

        # ==== 1) 距离 bias ====
        radial_bias = self.radial_mlp(radial)  # [E,1]
        radial_bias = radial_bias.squeeze(-1)  # [E]

        # ==== 2) 注意力 logit：q·k + radial_bias ====
        dot = (q_i * k_j).sum(dim=-1)  # [E]
        logits = dot / math.sqrt(self.hidden_nf) + radial_bias  # [E]

        if check_nan(logits):
            print("NaN in attention logits")
            # 不直接 exit，方便你继续打印更多信息
            # exit(-1)

        # ==== 3) scatter_softmax 按 row 分组 ====
        attn = scatter_softmax(logits, row, dim=0)  # [E]
        attn = attn.clamp_min(1e-9)  # 稍微稳定一点（避免 degenerate）

        if check_nan(attn):
            print("NaN in attention weights")

        # ==== 4) 边消息：m_ij = α_ij * v_j ====
        edge_feat = v_j * attn.unsqueeze(-1)  # [E, hidden_nf]

        if check_nan(edge_feat):
            print("NaN in edge_feat (after attention)")

        return edge_feat, attn

    # ===== node update =====

    def node_model(self, h, edge_index, edge_feat):
        """
        h:         [N, C_in]  原始节点特征
        edge_feat: [E, hidden_nf]  边消息
        return:
          h_out: [N, output_nf]
          agg:   [N, hidden_nf]
        """
        row, col = edge_index
        N = h.size(0)
        De = edge_feat.size(-1)

        agg = torch.zeros(N, De, device=h.device, dtype=edge_feat.dtype)
        agg.index_add_(0, row, edge_feat)  # sum_j m_ij

        if self.node_agg == "mean":
            deg = torch.bincount(row, minlength=N).clamp_min(1).to(agg.dtype)
            agg = agg / deg.unsqueeze(-1)

        # 聚合后映射到输出通道
        upd = self.node_proj(agg)          # [N, output_nf]
        base = self.h_in_proj(h)           # [N, output_nf]

        gate = self.node_gate(upd)         # [N, output_nf]
        if self.residual:
            h_out = base + gate * upd
        else:
            h_out = gate * upd

        if check_nan(h_out):
            print("NaN in node_model output")

        return h_out, agg

    # ===== coord update =====

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        coord:     [N,3]
        coord_diff:[E,3]
        edge_feat: [E, hidden_nf]
        return:
          coord_out:[N,3]
        """
        row, col = edge_index
        N = coord.size(0)

        if check_nan(coord):
            print("NaN in coord at start of coord_model")

        # edge_feat -> scalar weight w_ij
        w = self.coord_mlp(edge_feat.float())  # [E,1]
        if w.dim() == 1:
            w = w.unsqueeze(-1)

        # trans_ij = w_ij * (x_i - x_j)
        trans = coord_diff * w.to(coord_diff.dtype)  # [E,3]

        # sum_j trans_ij
        agg = torch.zeros(N, 3, device=coord.device, dtype=coord.dtype)
        agg.index_add_(0, row, trans)

        if self.coords_agg == "mean":
            deg = torch.bincount(row, minlength=N).clamp_min(1).to(coord.dtype)
            agg = agg / deg.unsqueeze(-1)
        elif self.coords_agg == "sum":
            pass
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")

        coord_out = coord + agg

        if check_nan(coord_out):
            print("NaN in coord_out")

        return coord_out

    # ===== forward =====

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        batch_size=1,
        k=30,
    ):
        """
        与 E_GCL_RM_Node 相同的前向接口。

        h:
          - dict 时：fusion["hidden"] 形状 [B,L,C] 或 [N,C]
          - tensor 时：[B,L,C] 或 [N,C]
        coord: [N,3]
        edge_index: (row, col)
        """
        # ------- 1) 处理融合 dict / 形状 --------
        if isinstance(h, dict):
            fusion = h
            h = fusion["hidden"]  # [B,L,C] or [N,C]
            edge_attr = fusion.get("edge_attr", edge_attr)
            node_attr = fusion.get("node_attr", node_attr)
            padding_mask = fusion.get("padding_mask", None)
            h = h.contiguous()

        # 兼容 [B,L,C] / [N,C]
        if h.dim() == 3:
            B, L, C = h.shape
            h = h.view(B * L, C).contiguous()
        elif h.dim() == 2:
            # [N,C]
            pass
        else:
            raise ValueError(f"Unsupported h.dim() = {h.dim()}")

        if check_nan(coord):
            print("NaN in input coord to SE3TransformerLayer")

        # ------- 2) 几何项：r^2 & coord_diff --------
        radial, coord_diff = self.coord2radial(edge_index, coord)
        if check_nan(coord_diff):
            print("NaN in coord_diff after coord2radial")

        # ------- 3) Q/K/V --------
        q = self.to_q(h)  # [N, hidden_nf]
        k = self.to_k(h)
        v = self.to_v(h)

        # ------- 4) Attention on edges --------
        edge_feat, attn = self.edge_model(q, k, v, radial, edge_index)
        if check_nan(edge_feat):
            print("NaN in edge_feat after edge_model")

        # ------- 5) Coord update --------
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        # ------- 6) Node update --------
        h, agg = self.node_model(h, edge_index, edge_feat)

        # 输出形状保持与原 EGNN 一致：h 为 [N,C_out]（外面若需要 [B,L,C] 自己再 reshape）
        return h, coord, edge_attr
