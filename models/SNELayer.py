import sys
sys.path.append("..")
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.EGNN import E_GCL_RM_Node
from models.SE3Transformer import SE3TransformerLayer


def _knn_edges_torch(coords_b_l_3: torch.Tensor, k: int, padding_mask: Optional[torch.Tensor] = None) \
    -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    纯 torch kNN 构图（逐 batch 独立，按欧氏距离，排除自环）
    输入:
      coords_b_l_3: [B, L, 3]
      k           : 近邻数（最终每个点连向 k 个邻居；若 L-1 < k 则取 L-1）
    返回:
      (row, col): 两个 LongTensor，一维 [E]，为拼接后的“全局索引边”；全局索引 = b*L + i
    """
    assert coords_b_l_3.dim() == 3 and coords_b_l_3.size(-1) == 3, "coords 必须是 [B, L, 3]"
    device = coords_b_l_3.device
    B, L, _ = coords_b_l_3.shape

    coords = coords_b_l_3
    coords = torch.where(torch.isinf(coords), torch.zeros_like(coords), coords)
    coords = torch.where(torch.isnan(coords), torch.zeros_like(coords), coords)

    if L <= 1 or k <= 0:
        row = torch.empty(0, dtype=torch.long, device=device)
        col = torch.empty(0, dtype=torch.long, device=device)
        return row, col

    k_eff = min(int(k), L - 1)
    rows, cols = [], []
    for b in range(B):
        C = coords[b]                                # [L,3]
        x2 = (C ** 2).sum(dim=1, keepdim=True)       # [L,1]
        dist = x2 + x2.t() - 2.0 * (C @ C.t())       # [L,L]
        dist = torch.clamp(dist, min=0.0)
        dist.fill_diagonal_(math.inf)                # 排除自环
        _, nn_idx = torch.topk(dist, k=k_eff, largest=False, dim=-1)  # [L,k]
        off = b * L
        rows.append(torch.arange(L, device=device).unsqueeze(1).expand(L, k_eff).reshape(-1) + off)
        cols.append(nn_idx.reshape(-1) + off)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    return row, col

class SpatialNeighborhoodEquivariantLayer(nn.Module):
    """
    轻量配体几何层：
      - 输入：原子 5 维特征 [B,L,5] 与 3D 坐标 [B,L,3]（可含 padding_mask）
      - 过程：kNN 构图 → 堆叠若干 E_GCL_RM_Node → 更新 (h, coords)
      - 输出：固定维度特征（全局池化+投影），同时返回节点级 (h, coords)
    不包含打分头。

    Args:
        in_node_nf : 输入原子特征维度（默认 5）
        hidden_nf  : EGNN 隐藏维
        out_dim    : 输出固定维度
        n_layers   : EGNN 层数
        k          : kNN 近邻数
        attention  : 是否使用边注意力
        coords_agg : 'sum' 或 'mean'，坐标聚合方式
        normalize  : 是否单位化坐标方向
        tanh       : 坐标增量是否接 Tanh
        device     : 设备
        pooling    : 'mean' 或 'sum'，全局池化方式
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.device = cfg.get('device', 'cpu')
        self.in_node_nf = cfg.get('in_node_nf')
        self.hidden_nf = cfg.get('hidden_nf')
        self.out_dim = cfg.get('out_dim')
        self.n_layers = cfg.get('n_layers')
        self.k = cfg.get('k')
        self.pooling = cfg.get('pooling')
        self.equivariant_model = cfg.get("equivariant_model", 'egnn')

        # 原子 5 维特征嵌入到隐藏维
        self.embedding_in = nn.Linear(self.in_node_nf, self.hidden_nf)

        # 堆叠 EGNN 层（传入 cfg 以设置 device，满足你的 E_GCL_RM_Node.__init__ 签名）
        if self.equivariant_model is None or self.equivariant_model == 'egnn':
            self.blocks = nn.ModuleList([
                E_GCL_RM_Node(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=0,
                    act_fn=nn.SiLU(),
                    residual=True,
                    attention=cfg.get('attention'),
                    normalize=cfg.get('normalize'),
                    coords_agg=cfg.get('coords_agg'),
                    tanh=cfg.get('tanh'),
                    cfg={"device": self.device},          # 关键：为你的实现传 device
                )
                for _ in range(self.n_layers)
            ])
        elif self.equivariant_model == 'se3transformer':
            self.blocks = nn.ModuleList([
                SE3TransformerLayer(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=0,
                    act_fn=nn.SiLU(),
                    residual=True,
                    attention=cfg.get('attention'),
                    normalize=cfg.get('normalize'),
                    coords_agg=cfg.get('coords_agg'),
                    tanh=cfg.get('tanh'),
                    cfg={"device": self.device},          # 关键：为你的实现传 device
                )
                for _ in range(self.n_layers)
            ])


        # 全局池化后再线性投影到固定维度
        self.readout = nn.Sequential(
            nn.LayerNorm(self.hidden_nf),
            nn.Linear(self.hidden_nf, self.out_dim),
        )

        self.to(self.device)

    def _global_pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        h: [B, L, H]
        mask: [B, L] (True=padding) or None
        """
        if mask is None:
            return h.sum(dim=1) if self.pooling == "sum" else h.mean(dim=1)

        valid = (~mask).float().unsqueeze(-1)       # [B,L,1]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)     # [B,1,1]
        summed = (h * valid).sum(dim=1)             # [B,H]
        return summed if self.pooling == "sum" else (summed / denom.squeeze(2))

    @torch.no_grad()
    def _sanitize_coords(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x

    def forward(
        self,
        atom_feat_5d: torch.Tensor,              # [B,L,5]
        coords_3d: torch.Tensor,                 # [B,L,3]
        padding_mask: Optional[torch.Tensor] = None,   # [B,L] True=padding
    ):
        """
        OUT:
        feat: （全局配体表征）  形状：[B, out_dim]
        h_out: （原子级特征）   形状：[B, L, H]
        x_out: （原子级坐标）   形状：[B, L, 3]
        """

        B, L, _ = atom_feat_5d.shape

        # 1) 输入嵌入
        h = self.embedding_in(atom_feat_5d.float().to(self.device))  # [B,L,H]
        x_input = coords_3d.to(self.device)                          # 保留原始坐标供输出使用
        x = self._sanitize_coords(x_input)                           # [B,L,3] —— 防止 NaN/Inf 影响计算

        # 2) kNN 构图（用未展平的 B×L×3 坐标来算邻居）

        row, col = _knn_edges_torch(x, k=min(self.k, L), padding_mask=padding_mask)     # [E], [E]
        edge_index = (row, col)

        # 3) 展平坐标到 [B*L,3]，但特征以 dict 形式传入以满足你的 forward
        x_flat = x.reshape(-1, 3)                    # [B*L,3]

        fusion = {
            "hidden": h,                             # [B,L,H] —— 你的 forward 会 reshape 并读取
            "padding_mask": padding_mask,            # 可选，用于下游需要
            # "edge_attr": None, "node_attr": None   # 按需补充
        }

        # 4) 逐层 EGNN 更新
        for block in self.blocks:
            h_flat, x_flat, _ = block(
                fusion,                  # h 以 dict 传入，匹配你的 E_GCL_RM_Node.forward
                edge_index=edge_index,
                coord=x_flat,            # [B*L,3]
                edge_attr=None,
                node_attr=None,
                batch_size=B,
                k=self.k,
            )
            # 回写 fusion["hidden"] 供下一层使用（你的实现会读取并再展平）
            fusion["hidden"] = h_flat.view(B, L, self.hidden_nf)

        # 5) 还原形状
        h_out = fusion["hidden"]                    # [B,L,H]
        # 返回原始配体坐标，避免在 SNE 层中修改后影响下游
        x_out = x_input                             # [B,L,3]

        # 6) 全局池化 → 固定维度向量
        pooled = self._global_pool(h_out, padding_mask)   # [B,H]
        feat = self.readout(pooled)                       # [B,out_dim]

        return feat, h_out, x_out, padding_mask
