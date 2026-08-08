import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_scatter import scatter_softmax, scatter_add, scatter_max

logger = logging.getLogger(__name__)

class E_GCL_RM_Node(nn.Module):
    """
    概述（Class）：
        E_GCL_RM_Node 实现了一个 E(n)-等变的图卷积层（EGNN 风格的 message passing 单元）。
        该层将节点特征与几何信息（节点坐标差/径向距离）结合，完成：
          1) 边特征/消息计算（edge_model）：由源/宿节点特征与径向项构造边消息，可选注意力归一化；
          2) 坐标更新（coord_model）：将边消息映射为标量，沿边方向矢量叠加，更新节点坐标；
          3) 节点更新（node_model）：聚合 K 邻边消息，门控残差方式更新节点特征。
        “RM_Node”命名表明“该实现移除了节点级 MLP 的显式变换”，只保留门控残差（gate * agg）。

        输入形状约定：
          - h:         [B*L, C]     （节点特征，C=input_nf）
          - coord:     [B*L, 3]     （节点坐标）
          - edge_index:(row, col)   （边索引，每条边从 col -> row）
          - batch_size: int         （批大小 B，用于把 E= B*N*K 的边数量 reshape 回 [B, N, K]）
          - k:         int          （每个节点的 K 个近邻）

        输出：
          - h_out:     [B*L, C]     （更新后的节点特征）
          - coord_out: [B*L, 3]     （更新后的坐标）
          - edge_attr: 任意        （原样返回，便于与框架接口对齐）
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, cfg=None):
        super(E_GCL_RM_Node, self).__init__()        # 初始化父类
        self.device = torch.device(cfg.get('device', 'cpu'))
        input_edge = input_nf * 2                     # 边 MLP 的节点特征输入维度（源+宿拼接）
        self.residual = residual                      # 是否使用节点残差更新
        self.attention = attention                    # 是否在边消息上使用注意力归一化
        self.normalize = normalize                    # 坐标方向是否单位化（除以长度）
        self.coords_agg = coords_agg                  # 坐标更新时邻居聚合方式：'sum' 或 'mean'
        self.tanh = tanh                              # 坐标增量是否经过 tanh 限幅
        self.epsilon = 1e-8                           # 数值稳定的小常数，避免除零
        edge_coords_nf = 1                            # 几何径向项维度（此处为 r^2 单标量）

        # self.linear1 = nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf)
        # print(type(input_edge))
        # print(type(edge_coords_nf))
        # print(type(edges_in_d))
        # print(type(hidden_nf))

        self.edge_mlp = nn.Sequential(                # 边消息 MLP：构造 m_ij
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),  # 输入：[h_i||h_j||r^2||edge_attr]
            act_fn,                                    # 激活函数
            nn.Linear(hidden_nf, hidden_nf),           # 映射到隐藏维度
            act_fn)                                    # 激活函数

        layer = nn.Linear(hidden_nf, 1, bias=False)   # 坐标标量头（hidden_nf -> 1）
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # 小增益初始化，避免过大位移

        coord_mlp = []                                # 构建坐标 MLP 列表
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))  # hidden -> hidden
        coord_mlp.append(act_fn)                      # 激活
        coord_mlp.append(layer)                       # hidden -> 1（标量系数）
        if self.tanh:                                 # 如果需要限幅
            coord_mlp.append(nn.Tanh())               # 使用 Tanh 限制标量范围
        self.coord_mlp = nn.Sequential(*coord_mlp)    # 打包为顺序网络

        self.node_gate = nn.Sequential(               # 节点门控：控制聚合量注入强度
            nn.Linear(hidden_nf, hidden_nf),          # hidden -> hidden
            nn.ReLU(),                                # 激活
            nn.Linear(hidden_nf, hidden_nf),          # hidden -> hidden
            nn.Sigmoid()                              # (0,1) 门
        )
        # self.node_gate = nn.Sequential(             # 更“深”的门控可替代（保留以便试验）
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.ReLU(),
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.ReLU(),
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.Sigmoid()
        # )

        if self.attention:                            # 如果启用注意力
            self.att_mlp = nn.Sequential(             # 边注意力打分器（标量）
                nn.Linear(hidden_nf, 1))              # hidden -> 1
        self.use_encoder_embed = False
        self.use_entropy_feat = False

    def edge_model(self, source, target, radial, edge_attr, batch_size, k, row, col):
        """
        概述（edge_model）：
            基于源/宿节点特征与几何径向项 r^2（可选拼接额外边特征），计算边消息 m_ij。
            若启用 attention，则对每个节点的 K 条入边做 softmax 归一化以形成注意力加权。
        参数：
            source:    [E, C]    源节点特征（h_i）
            target:    [E, C]    宿节点特征（h_j）
            radial:    [E, 1]    r^2（坐标差的平方范数）
            edge_attr: [E, D_e]  额外边特征（未使用时可为 None）
            batch_size:int       批大小 B（用于把 E 拆回 [B, N, K]）
            k:         int       每节点邻居数 K
        返回：
            out:       [E, hidden_nf]  边级消息特征 m_ij
        """
        # if edge_attr is None:  # Unused.
        #     out = torch.cat([source, target, radial], dim=1).to(device)
        # else:
        #     out = torch.cat([source, target, radial, edge_attr], dim=1).to(device)
        # computing m_{ij}

        # print('source ', source.shape)
        # print('target ', target.shape)
        # print('radial ', radial.shape)

        if check_nan(source):
            print('NaN in source at start of edge_model')

        if check_nan(target):
            print('NaN in target at start of edge_model')

        if check_nan(radial):
            print('NaN in radial at start of edge_model')

        out = torch.cat([source, target, radial], dim=1).to(self.device)  # 拼接源/宿特征与 r^2
        out = self.edge_mlp(out.float())                              # 通过边 MLP 得到边消息
        # need to use softmax to normalize
        # if self.attention:                                            # 若启用注意力
        #     attn = self.att_mlp(out)                                  # 计算注意力打分 [E,1]
        #     # print('attn : ', attn.shape)
        #     att_val = torch.softmax(attn.view(batch_size, -1, k), dim=-1).view(-1, 1)  # 对 K 维归一化
        #     out = out * att_val                                       # 用注意力权重缩放边消息

        if check_nan(out):
            print('NaN in out at end of edge_model')

        if self.attention:
            attn = self.att_mlp(out).squeeze(-1)  # [E]

            # torch_scatter 的分组 softmax

            # print('attn device : ', attn.device)
            # print('row device : ', row.device)
            # print('d1vice: ', self.device)


            att_val_flat = scatter_softmax(attn, row, dim=0)        # [E]
            att_val = att_val_flat.unsqueeze(-1)                    # [E,1]

            if check_nan(att_val):
                print('NaN in att_val at end of edge_model')
                exit(-1)

            out = out * att_val                                             # 加权边消息



        return out                                                    # 返回边消息

    # def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
    #     """
    #     概述（node_model）：
    #         将边消息按每个节点聚合（对 K 邻居求和），得到聚合量 agg；
    #         使用门控残差方式：h_out = x + gate(agg) * agg，更新节点特征。
    #     参数：
    #         x:         [B*L, C]         节点特征
    #         edge_index:(row, col)       边索引
    #         edge_attr: [E, hidden_nf]   边消息特征（来自 edge_model）
    #         node_attr: 任意             预留参数（本实现未使用）
    #         batch_size:int              批大小 B
    #         k:         int              邻居数 K
    #     返回：
    #         out:       [B*L, C]         更新后的节点特征
    #         agg:       [B*L, hidden_nf] 聚合后的邻边消息（可用于可视化/调试）
    #     """
    #     row, col = edge_index                                  # 取出边索引
    #     # row = row.to(device)
    #     dim = edge_attr.size(-1)                               # 记录边消息维度 hidden_nf
    #     edge_attr = edge_attr.view(batch_size, -1, k, dim)     # 形状重构为 [B, N, K, hidden_nf]
    #     agg = torch.sum(edge_attr, dim=2).view(-1, dim)        # 对 K 邻居求和 → [B*N, hidden_nf]
    #     out = x                                                # 初始设为原节点特征
    #     if self.residual:                                      # 若启用残差
    #         out = x + self.node_gate(agg) * agg                # 使用门控残差更新
    #     return out, agg                                        # 返回更新后的节点特征与聚合量

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        """
        x:         [BN, C]
        edge_index:(row, col)
        edge_attr: [E, De]  —— 已是 edge_model 输出（可含注意力权重）
        返回:
        out: [BN, C]
        agg: [BN, De]  —— 聚合后的邻边消息（未投影，便于可视化）
        """
        row, col = edge_index
        BN, C    = x.size()
        De       = edge_attr.size(-1)

        # 1) 边→节点聚合（按 row 分组；若你想按入边聚合就改成 col）
        agg = torch.zeros(BN, De, device=x.device, dtype=edge_attr.dtype)
        agg.index_add_(0, row, edge_attr)              # sum 聚合

        # 2) 可选：均值归一化
        if getattr(self, "node_agg", "sum") == "mean":
            deg = torch.bincount(row, minlength=BN).clamp_min(1)  # [BN]
            agg = agg / deg.to(agg.dtype).unsqueeze(-1)

        # 3) 维度对齐：如果 De != C，需要一个投影层把 agg 映到节点通道 C
        if De != C:
            # 需要在 __init__ 里定义：
            #   self.node_proj = nn.Linear(De, C, bias=False)
            #   self.node_gate = nn.Sequential(nn.Linear(C, C), nn.Sigmoid())
            upd = self.node_proj(agg)                  # [BN, C]
            gate_in = upd
        else:
            upd = agg                                  # [BN, C]
            # 若已有 self.node_gate: hidden_nf->hidden_nf，则维度已对齐
            gate_in = upd

        # 4) 门控残差
        out = x
        if getattr(self, "residual", True):
            if hasattr(self, "node_gate"):
                # print("Using node_gate for gating in node_model")
                gate = torch.sigmoid(self.node_gate(gate_in))  # [BN, C]
                # print("[DEBUG] node_gate out requires_grad:", gate.requires_grad)
            else:
                # 没 gate 就当恒等门（全部通过）
                gate = torch.ones_like(upd)
            out = x + gate * upd

        return out, agg

    # def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
    #     """
    #     概述（coord_model）：
    #         使用边消息经坐标 MLP 得到标量系数，逐边与方向向量 coord_diff 相乘，得到边级坐标增量；
    #         然后在邻居维聚合（sum/mean），以残差形式更新节点坐标。
    #     参数：
    #         coord:     [B*L, 3]         节点坐标
    #         edge_index:(row, col)       边索引
    #         coord_diff:[E, 3]           方向向量（coord[row] - coord[col]）
    #         edge_feat: [E, hidden_nf]   边消息特征
    #         batch_size:int              批大小 B
    #         k:         int              邻居数 K
    #     返回：
    #         coord:     [B*L, 3]         更新后的坐标
    #     """
    #     row, col = edge_index                                  # 取边索引（此处未直接使用，但保留一致性）
    #     # row = row.to(device)
    #     coord_diff = coord_diff.to(self.device)                # 将方向向量移至目标设备
    #     edge_feat_scalar = self.coord_mlp(edge_feat)

    #     # print('edge_feat_scalar', edge_feat_scalar.shape)
    #     # print('coord_diff', coord_diff.shape)

    #     trans = coord_diff *  edge_feat_scalar                 # 标量 * 方向 → 边级坐标增量 [E,3]
    #     trans = trans.view(batch_size, -1, k, 3)               # 重构为 [B, N, K, 3]
    #     # print('trans : ', trans.shape)
    #     if self.coords_agg == 'sum':                           # 若按和聚合
    #         agg = torch.sum(trans, dim=2).view(-1, 3)          # 对 K 聚合 → [B*N,3]
    #         # agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
    #     elif self.coords_agg == 'mean':                        # 若按均值聚合
    #         agg = torch.mean(trans, dim=2).view(-1, 3)         # 对 K 聚合 → [B*N,3]
    #         # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
    #     else:                                                  # 非法配置抛错
    #         raise Exception('Wrong coords_agg parameter' % self.coords_agg)
    #     coord = coord + agg                                    # 残差式更新坐标
    #     return coord                                           # 返回更新后的坐标
    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
        """
        使用边消息得到标量/向量系数，逐边与方向向量相乘 → 边级坐标增量；
        再按节点（edge_index 的一端）聚合，做残差更新。
        兼容可变邻居数、双向边、padding/mask（mask 请在上游过滤边）。
        参数中 batch_size, k 不再需要；
        """
        row, col = edge_index                                   # row: 源节点索引, col: 宿节点索引
        BN = coord.size(0)                                      # 批展平后的节点数


        if check_nan(coord):
            print('NaN in coord at start of coord_model')

        if check_nan(coord_diff):
            print('NaN in coord_diff at start of coord_model')

        if check_nan(edge_feat):
            print('NaN in edge_feat at start of coord_model')


        # 1) 边级权重：支持输出 [E,1]（标量）或 [E,3]（逐维系数）
        w = self.coord_mlp(edge_feat.float())                  # [E,1] 或 [E,3]
        if w.dim() == 1:                                       # 罕见：被 squeeze 成 [E]
            w = w.unsqueeze(-1)
        if w.size(-1) == 1:
            trans = coord_diff * w                              # [E,3] = [E,3] * [E,1]
        elif w.size(-1) == 3:
            trans = coord_diff * w                              # [E,3] = [E,3] * [E,3]
        else:
            raise ValueError(f"coord_mlp must output 1 or 3 dims, got {w.size(-1)}")

        trans = trans.to(coord.dtype)

        # 2) 按节点聚合（默认聚合到 row，即更新源节点）
        agg = torch.zeros(BN, 3, device=coord.device, dtype=coord.dtype)  # [BN,3]

        if check_nan(agg):
            print('NaN in agg at coord_model before index_add')
            exit(-1)

        agg.index_add_(0, row, trans)                                     # sum 聚合

        if check_nan(agg):
            print('NaN in agg at coord_model after index_add')
            exit(-1)

        # 3) 归一化策略（sum / mean）
        if self.coords_agg == 'sum':
            pass  # 已经是 sum
        elif self.coords_agg == 'mean':
            deg = torch.bincount(row, minlength=BN).clamp_min(1)          # [BN]
            agg = agg / deg.to(agg.dtype).unsqueeze(-1)
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")

        if check_nan(agg):
            print('NaN in agg at coord_model before update')
            exit(-1)

        # 4) 残差更新
        coord = coord + agg                                               # [BN,3]

        if check_nan(coord):
            print('NaN in coord at end of coord_model')
            exit(-1)

        return coord

    def coord2radial(self, edge_index, coord):
        """
        概述（coord2radial）：
            基于坐标计算每条边的方向向量（i<-j，coord[row]-coord[col]）与其平方范数 r^2；
            若 normalize=True，则将方向向量单位化。
        参数：
            edge_index:(row, col)   边索引
            coord:     [B*L, 3]     节点坐标
        返回：
            radial:    [E, 1]       r^2 距离标量
            coord_diff:[E, 3]       方向向量
        """
        row, col = edge_index                                  # 取边两端索引
        coord_diff = coord[row] - coord[col]                   # 方向向量（j->i）
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(self.device)  # r^2，并扩展成 [E,1]
        if self.normalize:                                     # 若需要单位化方向
            norm = torch.sqrt(radial).detach() + self.epsilon  # 计算范数并加稳定项
            coord_diff = coord_diff / norm                     # 单位化方向向量
        return radial, coord_diff                              # 返回 r^2 与方向向量

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        概述（forward）：
            该前向过程依次完成：几何项计算 → 边消息计算 → 坐标更新 → 节点更新。
        参数：
            h:         [B*L, C]         节点特征（如 ESM 残基嵌入）
            edge_index:(row, col)       边索引（col -> row）
            coord:     [B*L, 3]         节点坐标
            edge_attr: 任意             额外边特征（本实现未使用，可扩展）
            node_attr: 任意             额外节点特征（本实现未使用，可扩展）
            batch_size:int              批大小 B（用于把 E 拆回 [B, N, K]）
            k:         int              邻居数 K
        返回：
            h_out:     [B*L, C]         更新后的节点特征
            coord_out: [B*L, 3]         更新后的节点坐标
            edge_attr: 任意             原样返回（与上游接口对齐）
        """



        if isinstance(h, dict):
            fusion = h
            h = fusion["hidden"]                       # [B,L,E] or [N,E]
            # 可按需取其它融合特征：
            edge_attr = fusion.get("edge_attr", edge_attr)
            node_attr = fusion.get("node_attr", node_attr)
            padding_mask = fusion.get("padding_mask", None)
            # 还要确保 h 是连续的 float tensor
            h = h.contiguous()

        B, L, C = h.shape
        h = h.reshape(B*L, C).contiguous()

        if check_nan(coord):
            print('NaN in input coord to E_GCL_RM_Node')
            exit(-1 )

        row, col = edge_index                                  # 取边两端索引（后续未直接用，但有时调试方便）
        radial, coord_diff = self.coord2radial(edge_index, coord)   # 计算 r^2 与方向向量
        # [B * L * 30], [B * L * 30, 3]

        if check_nan(coord_diff):
            print('NaN in coord_diff after coord2radial')
            exit(-1 )

        # print('h : ', h.shape)
        # print('row : ', row.shape)
        # print('col : ', col.shape)
        # print('h[row] : ', h[row].shape)
        # print(' h[col] : ', h[col].shape)
        # print('radial : ', type(radial))
        # print('edge_attr : ', type(edge_attr))
        # print('batch_size : ', type(batch_size))
        # print('k : ', type(k))


        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k, row, col)   # 计算边消息 m_ij
        # m_{ij}, [B * L * 30, hidden_nf]

        if check_nan(edge_feat):
            print('NaN in edge_feat after edge_model')
            exit(-1 )

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch_size, k)  # 坐标更新

        if check_nan(coord):
            print('NaN in coord after coord_model')
            exit(-1 )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)       # 节点更新（门控残差）

        return h, coord, edge_attr                         # 返回更新后的节点特征、坐标与边特征（原样）


def check_nan(x):
    x = x.detach()
    # 尽量少做 CUDA 运算；若你只是要诊断，可直接搬到 CPU 检
    try:
        if x.is_cuda:
            return (~torch.isfinite(x)).any().item()
        else:
            return (~torch.isfinite(x)).any().item()
    except RuntimeError:
        # 已经出现 device-side assert，返回 True 触发上游退出
        return True