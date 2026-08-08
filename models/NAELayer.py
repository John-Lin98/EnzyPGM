import sys
sys.path.append("..")

import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RFFLayer import ResidueFunctionFusionLayer
from models.EGNN import E_GCL_RM_Node
from models.SE3Transformer import SE3TransformerLayer

import logging

logger = logging.getLogger(__name__)

def _knn_edges_torch(coords_b_l_3: torch.Tensor, k: int, padding_mask=None):
    """
    纯 torch kNN 构图（逐 batch 独立，按欧氏距离，排除自环）
    输入:
      coords_b_l_3: [B, L, 3]  坐标（可含 NaN/Inf，会自动置 0）
      k           : 近邻数（最终每个点连向 k 个邻居；若 L-1 < k 则取 L-1）
    返回:
      (row, col): 两个 LongTensor，一维 [E]，为拼接后的“全局索引边”
                  其中全局索引 = b*L + i
    """
    assert coords_b_l_3.dim() == 3 and coords_b_l_3.size(-1) == 3, "coords 必须是 [B, L, 3]"

    device = coords_b_l_3.device
    B, L, _ = coords_b_l_3.shape

    # 清洗非法值
    coords = coords_b_l_3
    coords = torch.where(torch.isinf(coords), torch.zeros_like(coords), coords)
    coords = torch.where(torch.isnan(coords), torch.zeros_like(coords), coords)

    rows, cols = [], []

    if L <= 1 or k <= 0:
        # 没有有效边
        row = torch.empty(0, dtype=torch.long, device=device)
        col = torch.empty(0, dtype=torch.long, device=device)
        return row, col

    k = int(k)
    k_eff = min(k, L - 1)  # 排除自环后最多 L-1 个邻居

    for b in range(B):
        C = coords[b]                     # [L, 3]

        # 计算 pairwise 距离的平方（更稳更快，不开 sqrt）
        # dist_ij = ||xi - xj||^2 = (xi^2 + xj^2 - 2 xi·xj)
        x2   = (C ** 2).sum(dim=1, keepdim=True)   # [L,1]
        dist = x2 + x2.t() - 2.0 * (C @ C.t())     # [L,L]
        # 数值稳定：可能有极小负数，截断为 >=0
        dist = torch.clamp(dist, min=0.0)

        # 排除自环
        dist.fill_diagonal_(math.inf)

        # 取每个点的 k_eff 个最近邻
        nn_dist, nn_idx = torch.topk(dist, k=k_eff, largest=False, dim=-1)  # [L, k_eff]

        # 构建边（行 i -> 列 nn_idx[i, j]）
        row_b = torch.arange(L, device=device).unsqueeze(1).expand(L, k_eff).reshape(-1)  # [L*k_eff]
        col_b = nn_idx.reshape(-1)                                                        # [L*k_eff]

        # 打到全局索引（batch 偏移）
        off = b * L
        rows.append(row_b + off)
        cols.append(col_b + off)

    row = torch.cat(rows, dim=0) if rows else torch.empty(0, dtype=torch.long, device=device)
    col = torch.cat(cols, dim=0) if cols else torch.empty(0, dtype=torch.long, device=device)
    return row, col
# def _knn_edges_torch(coords_b_l_3, k, padding_mask=None):
#     # coords_b_l_3: [B,L,3], padding_mask: [B,L] (True=pad)
#     device = coords_b_l_3.device
#     B, L, _ = coords_b_l_3.shape
#     rows, cols = [], []
#     k_eff = min(int(k), L - 1)

#     for b in range(B):
#         C = coords_b_l_3[b]                          # [L,3]
#         valid = torch.ones(L, dtype=torch.bool, device=device) if padding_mask is None else ~padding_mask[b]
#         idx = torch.nonzero(valid, as_tuple=False).flatten()   # 有效节点索引
#         if idx.numel() <= 1:
#             continue
#         Cv = C[idx]                                  # 只对有效节点构图
#         x2 = (Cv**2).sum(1, keepdim=True)
#         dist = x2 + x2.t() - 2.0 * (Cv @ Cv.t())
#         dist.clamp_(min=0.0)
#         dist.fill_diagonal_(math.inf)
#         k_b = min(k_eff, Cv.size(0) - 1)
#         _, nn_idx_local = torch.topk(dist, k=k_b, largest=False, dim=-1)  # [Nv, k_b]

#         row_local = torch.arange(Cv.size(0), device=device).unsqueeze(1).expand(-1, k_b).reshape(-1)
#         col_local = nn_idx_local.reshape(-1)
#         # 映射回全局 L 维度索引
#         row_b = idx[row_local]
#         col_b = idx[col_local]
#         off = b * L
#         rows.append(row_b + off)
#         cols.append(col_b + off)

#     row = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)
#     col = torch.cat(cols) if cols else torch.empty(0, dtype=torch.long, device=device)
#     return row, col



class NAELayer(nn.Module):
    """
    NAELayer：在本层内
      1) 计算 EC(1..4) 嵌入并作为条件向量 cond_add 注入到 ESM 的每一层；
      2) 逐层调用 RFFL（分层前向）完成语言侧编码；
      3) 使用 EGNN 对坐标做若干步几何更新（融合 RFFL 输出的 hidden/prob/padding_mask）。
    cfg（dict 或带属性对象）字段示例：
      - esm_path: str
      - ec_vocab_sizes: (9, 75, 256, 3157)
      - egnn_layers: int = 3
      - egnn_hidden_nf: Optional[int]
      - egnn_attention: bool = True
      - egnn_coords_agg: str = "mean"
      - egnn_normalize: bool = False
      - egnn_tanh: bool = False
      - k: int = 30
      - egnn_steps: int = egnn_layers
      - randwalk_step: float = 3.75
      - fuse_use_encoder_embed: bool = True
      - fuse_use_entropy_feat: bool = True
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cpu'))

        self.equivariant_model = cfg.get('equivariant_model', 'egnn')

        def _get(name, default=None):
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            else:
                raise Exception('config is not dict')

        # 1) RFFL：仅负责 ESM 分层前向（不做 EC）
        self.rffl = ResidueFunctionFusionLayer(cfg=cfg)  # 直接把 cfg 传进去
        self.rffl.to(self.device)

        # 嵌入维度
        self.embed_dim = self.rffl.embed_dim  # 来自 ESM encoder

        # print('self.rffl.embed_dim', self.rffl.embed_dim)

        # 2) 在 NAELayer 中定义 EC 1..4 的嵌入
        ec_vocab_sizes = _get("ec_vocab_sizes", (9, 75, 256, 3157))
        ec1_size, ec2_size, ec3_size, ec4_size = ec_vocab_sizes
        E = self.embed_dim
        self.ec1_embeddings = nn.Embedding(ec1_size, E).to(self.device)
        self.ec2_embeddings = nn.Embedding(ec2_size, E).to(self.device)
        self.ec3_embeddings = nn.Embedding(ec3_size, E).to(self.device)
        self.ec4_embeddings = nn.Embedding(ec4_size, E).to(self.device)

        nn.init.normal_(self.ec1_embeddings.weight, mean=0.0, std=E ** -0.5)
        nn.init.normal_(self.ec2_embeddings.weight, mean=0.0, std=E ** -0.5)
        nn.init.normal_(self.ec3_embeddings.weight, mean=0.0, std=E ** -0.5)
        nn.init.normal_(self.ec4_embeddings.weight, mean=0.0, std=E ** -0.5)

        # 3) EGNN 堆叠
        egnn_hidden_nf = _get("egnn_hidden_nf", self.embed_dim)

        # print('egnn_hidden_nf', egnn_hidden_nf, type(egnn_hidden_nf))

        egnn_layers = int(_get("egnn_layers", 3))
        self.k = int(_get("k", 30))

        if self.equivariant_model is None or self.equivariant_model == "egnn":
            self.egnn = nn.ModuleList([
                E_GCL_RM_Node(
                    input_nf=self.embed_dim,
                    output_nf=self.embed_dim,
                    hidden_nf=egnn_hidden_nf,
                    attention=bool(_get("egnn_attention", True)),
                    coords_agg=_get("egnn_coords_agg", "mean"),
                    normalize=bool(_get("egnn_normalize", False)),
                    tanh=bool(_get("egnn_tanh", False)),
                    cfg = cfg
                ).to(self.device)
                for _ in range(egnn_layers)
            ])
            self.egnn_steps = int(_get("egnn_steps", egnn_layers))
        elif self.equivariant_model == "se3transformer":
            self.egnn = nn.ModuleList([
                SE3TransformerLayer(
                    input_nf=self.embed_dim,
                    output_nf=self.embed_dim,
                    hidden_nf=egnn_hidden_nf,
                    attention=bool(_get("egnn_attention", True)),
                    coords_agg=_get("egnn_coords_agg", "mean"),
                    normalize=bool(_get("egnn_normalize", False)),
                    tanh=bool(_get("egnn_tanh", False)),
                    cfg = cfg
                ).to(self.device)
                for _ in range(egnn_layers)
            ])
            self.egnn_steps = int(_get("egnn_steps", egnn_layers))


        # 4) 随机游走步长
        self.randwalk_step = float(_get("randwalk_step", 3.75))

        # 5) RFFL 融合选项下发到 EGNN（是否融合 encoder_embedding/熵）
        if len(self.egnn) > 0 and hasattr(self.egnn[0], "use_encoder_embed"):
            use_enc = bool(_get("merge_encoder_embed", False))
            use_ent = bool(_get("merge_entropy_feat", False))
            for gcl in self.egnn:
                gcl.use_encoder_embed = use_enc
                gcl.use_entropy_feat  = use_ent

    # @torch.no_grad()
    # def _init_masked_coords(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    #     """对被 mask 的位置做随机球面步进初始化。"""
    #     B, L, _ = coords.shape
    #     coords = coords.clone()
    #     device = coords.device
    #     for i in range(B):
    #         for j in range(L):
    #             if mask[i, j] == 1:
    #                 theta = torch.empty(1, device=device).uniform_(0, math.pi)
    #                 phi   = torch.empty(1, device=device).uniform_(0, 2 * math.pi)
    #                 step = self.randwalk_step
    #                 base = coords[i, j - 1] if j > 0 else coords[i, j]
    #                 dx = step * torch.sin(theta) * torch.sin(phi)
    #                 dy = step * torch.sin(theta) * torch.cos(phi)
    #                 dz = step * torch.cos(theta)
    #                 coords[i, j, 0] = base[0] + dx
    #                 coords[i, j, 1] = base[1] + dy
    #                 coords[i, j, 2] = base[2] + dz
    #     return coords
    # @torch.no_grad()
    # @torch.cuda.amp.autocast(enabled=False)  # 这里用 fp32 更稳
    # def _init_masked_coords(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    #     """
    #     coords: [B,L,3] (float16/32/bf16)
    #     mask  : [B,L]   (bool 或 0/1)，True 表示要初始化的位置
    #     """
    #     # ===== 形状/设备/类型自检 =====
    #     assert coords.dim() == 3 and coords.size(-1) == 3, f"coords shape {coords.shape} invalid"
    #     B, L, _ = coords.shape
    #     device = coords.device
    #     dtype_in = coords.dtype

    #     mask = mask.to(device)
    #     if mask.dtype != torch.bool:
    #         # 兼容 0/1、byte、int 等；其他值直接视为 True
    #         mask = mask != 0
    #     assert mask.shape == (B, L), f"mask shape {mask.shape} != coords[:2] {coords.shape[:2]}"

    #     # ===== 向量化实现 =====
    #     # 基准点：j>0 用 j-1；j=0 用自身
    #     base = coords.clone()
    #     base[:, 1:, :] = coords[:, :-1, :]    # [B,L,3]

    #     # 生成随机单位向量
    #     v = torch.randn(B, L, 3, device=device, dtype=torch.float32)
    #     v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    #     step = float(getattr(self, "randwalk_step", 3.75))  # Å
    #     delta = step * v                                    # [B,L,3]

    #     # 仅对 mask 位置加扰动
    #     coords32 = coords.float().clone()
    #     coords32[mask] = base.float()[mask] + delta[mask]

    #     # 数值防护
    #     coords32 = torch.nan_to_num(coords32, nan=0.0, posinf=1e6, neginf=-1e6)
    #     return coords32.to(dtype_in)
    # @torch.no_grad()
    # def _init_masked_coords(self, coords, mask):
    #     with torch.cuda.amp.autocast(False):                # 关掉 autocast
    #         assert coords.dim()==3 and coords.size(-1)==3
    #         B, L, _  = coords.shape
    #         device    = coords.device
    #         dtype_in  = coords.dtype

    #         mask = mask.to(device)
    #         if mask.dtype is not torch.bool:                # 强转成干净的 bool
    #             mask = mask != 0
    #         assert mask.shape == (B, L)

    #         base  = coords.clone()
    #         base[:, 1:, :] = coords[:, :-1, :]

    #         v = torch.randn(B, L, 3, device=device, dtype=torch.float32)
    #         v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    #         step  = float(getattr(self, "randwalk_step", 3.75))
    #         delta = step * v                                # [B,L,3]

    #         coords32 = coords.float()
    #         mask3 = mask.unsqueeze(-1)                      # [B,L,1]

    #         # 任选其一（都安全）：
    #         # A) where
    #         new_vals = base.float() + delta
    #         coords32 = torch.where(mask3, new_vals, coords32)
    #         # B) 乘掩码（注释掉 A 再用 B 也行）
    #         # coords32 = coords32 + mask3.float() * (base.float() + delta - coords32)

    #         coords32 = torch.nan_to_num(coords32, nan=0.0, posinf=1e6, neginf=-1e6)
    #         return coords32.to(dtype_in)

    @torch.no_grad()
    def _init_masked_coords(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        对 mask==True 的位置做随机球面步进初始化；其余保持不变。
        更稳健：
        - 禁止 autocast（fp32 计算随机量与赋值）
        - 严格校验 mask 形状/类型/有限性
        - 显式 expand 掩码到 [B,L,3]，避免隐式广播
        - 对非有限坐标做清洗与范围钳位
        """
        with torch.cuda.amp.autocast(False):
            # ---- 形状/设备检查 ----
            assert coords.dim() == 3 and coords.size(-1) == 3, f"coords shape {coords.shape}"
            B, L, _  = coords.shape
            device    = coords.device
            dtype_in  = coords.dtype

            # ---- 掩码清洗：到同设备、转为干净 bool、去除非有限值 ----
            if mask.device != device:
                mask = mask.to(device)
            if mask.dtype != torch.bool:
                # 若 mask 含 NaN/±Inf，强制置 False；否则按 !=0 转 bool
                if not torch.isfinite(mask).all():
                    mask = torch.zeros(B, L, device=device, dtype=torch.bool)
                else:
                    mask = (mask != 0)
            assert mask.shape == (B, L), f"mask shape {mask.shape} != {(B, L)}"

            # 无需初始化直接返回
            if not mask.any():
                return coords

            # ---- 坐标到 fp32 + 非有限清洗（避免把 NaN 传播）----
            coords32 = coords.float()
            coords32 = torch.where(torch.isfinite(coords32), coords32, torch.zeros_like(coords32))

            # ---- 基准点：j>0 用 j-1；j=0 用自身 ----
            base = coords32.clone()
            base[:, 1:, :] = coords32[:, :-1, :]

            # ---- 随机单位向量 * 步长 ----
            v = torch.randn(B, L, 3, device=device, dtype=torch.float32)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            step   = float(getattr(self, "randwalk_step", 3.75))   # Å
            newval = base + step * v                               # [B,L,3]

            # ---- 显式扩展掩码到 [B,L,3]，避免隐式广播 ----
            mask3 = mask.view(B, L, 1).expand_as(coords32).contiguous()

            # ---- 按位选择（不使用布尔索引写入）----
            coords32 = torch.where(mask3, newval, coords32)

            # ---- 数值防护：NaN→0、∞→大常数，整体钳位到合理范围 ----
            coords32 = torch.nan_to_num(coords32, nan=0.0, posinf=1e6, neginf=-1e6)
            coords32 = torch.clamp(coords32, min=-1e3, max=1e3)    # ±1000 Å 安全边界

            return coords32.to(dtype_in)

    def _ec_cond_add(self, ec1, ec2, ec3, ec4, B: int) -> torch.Tensor:
        """计算 EC 条件向量并扩展为 [B,1,E]，供每层注入。"""
        E = self.embed_dim

        ec1_emb = self.ec1_embeddings(ec1.view(-1, 1)).view(B, 1, E)
        ec2_emb = self.ec2_embeddings(ec2.view(-1, 1)).view(B, 1, E)
        ec3_emb = self.ec3_embeddings(ec3.view(-1, 1)).view(B, 1, E)
        ec4_emb = self.ec4_embeddings(ec4.view(-1, 1)).view(B, 1, E)
        cond_add = ec1_emb + ec2_emb + ec3_emb + ec4_emb            # [B,1,E]
        return cond_add

    def _sanitize_hidden(self, x: torch.Tensor, name: str = "hidden", max_abs: float = 1e4):
        """
        对 RFFL 输出做一次数值清洗：
        - 将 NaN / ±Inf 替换为 0 / ±max_abs
        - 对绝对值特别大的有限值进行裁剪，避免后续层数值继续放大
        不会 detach，仍然保留梯度。
        """
        # 快速返回：完全正常就不做额外操作
        if torch.isfinite(x).all():
            return x

        with torch.no_grad():
            numel = x.numel()
            num_nan = torch.isnan(x).sum().item()
            num_posinf = torch.isinf(x).sum().item()
            num_neginf = torch.isinf(-x).sum().item()  # 统计负无穷
            max_val = torch.nan_to_num(x).abs().max().item()
            print(f"[sanitize] {name}: numel={numel} NaN={num_nan} +Inf={num_posinf} -Inf={num_neginf} max|x|={max_val:.3e}")

        # 1) 先把 NaN / Inf 替换掉
        x = torch.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs)
        # 2) 再把极端大的有限值裁剪一下
        x = x.clamp(min=-max_abs, max=max_abs)
        return x


    def forward(
        self,
        src_tokens: torch.LongTensor,     # [B,L]
        src_lengths: torch.LongTensor,    # [B]
        coors: torch.Tensor,              # [B,L,3]
        mask: torch.Tensor,               # [B,L] 1=masked
        ec1: torch.LongTensor,            # [B]
        ec2: torch.LongTensor,            # [B]
        ec3: torch.LongTensor,            # [B]
        ec4: torch.LongTensor,            # [B]
        padding_mask,
    ) -> Dict[str, torch.Tensor]:

        B, L = src_tokens.size(0), src_tokens.size(1)
        src_tokens = src_tokens.to(self.device)
        src_lengths = src_lengths.to(self.device)
        coors = coors.to(self.device)
        mask = mask.to(self.device)
        ec1 = ec1.to(self.device); ec2 = ec2.to(self.device)
        ec3 = ec3.to(self.device); ec4 = ec4.to(self.device)


        if check_nan(coors):
            print('NaN in input coors to NAE')
            exit(-1 )

        # ---------- 1) 计算 EC 条件 ----------
        cond_add = self._ec_cond_add(ec1, ec2, ec3, ec4, B)  # [B,1,E]

        x_in = None                           # 第一层没有上一层输出
        last_encoder_embedding = None         # 记录“送入最后一层前”的表示
        final_hidden = None                   # 记录最后一层输出
        final_logits = None
        final_prob = None

        # ---------- 2) 坐标初始化（对 mask 位随机游走） ----------
        coords = self._init_masked_coords(coors, mask)       # [B,L,3]

        if check_nan(coords):
            print('NaN in coords after init_masked_coords')
            print(coords)
            exit(-1 )


        # print('coords 1 : ', coords)
        for layer_idx in range(self.rffl.num_layers):

            # print(f'layer : {layer_idx} start')

            # print('before rffl ')

            out = self.rffl(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                coors=coors,                 # 未使用，占位
                mask=mask,
                layer_idx=layer_idx,         # 指定只跑这一层
                x_in=x_in,                   # 前一层输出（首层为 None）
                cond_add=cond_add,           # 注入 EC 条件（广播到全序列）
                padding_mask=padding_mask
            )

            # ========= 新增：清洗 RFFL 输出的 hidden，避免 NaN/Inf 传入 EGNN =========
            hidden = out["hidden"]          # [B,L,E]
            hidden = self._sanitize_hidden(hidden, name=f"rffl_hidden_layer{layer_idx}")
            out["hidden"] = hidden          # 回写，确保后续用的是清洗后的结果


            # 本层输出作为下一层输入
            x_in = out["hidden"]             # [B,L,E]
            padding_mask = out["padding_mask"]
            last_encoder_embedding = out["encoder_embedding"]  # 进入该层前的表示（含 cond_add）

            if check_nan(coords):
                print('NaN in coords after rffl')
                print(coords)
                exit(-1 )

            # print('rfnn finish : out["hidden"]   : ', out["hidden"].shape)

            if (layer_idx + 1) % 11 == 0:
                final_hidden = out["hidden"]                 # [B,L,E]
                final_logits = out["logits"]                 # [B,L,V]
                final_prob   = out["prob"]                   # [B,L,V]
                fusion_out = {
                    "hidden": final_hidden,               # [B,L,E]
                    "prob": final_prob,                   # [B,L,V]
                    "padding_mask": padding_mask,         # [B,L]
                    "encoder_embedding": last_encoder_embedding,  # [B,L,E]
                }
                # ---------- 3) EGNN 更新（融合 RFFL 输出） ----------
                coords_flat = coords.reshape(-1, 3)       # [B*L,3]

                egnn_layer_idx = int(layer_idx / 11)
                row, col = _knn_edges_torch(coords, k=self.k, padding_mask=padding_mask)      # (row,col): [E]
                edge_index = (row, col)

                # print(f'egnn : {egnn_layer_idx} start')

                if check_nan(coords_flat):
                    print('NaN in coords_flat before egnn')
                    print(coords_flat)
                    exit(-1 )

                h_flat, coords_flat, _ = self.egnn[egnn_layer_idx](
                    fusion_out, edge_index, coords_flat,
                    edge_attr=None, node_attr=None,
                    batch_size=B, k=self.k
                )

                if check_nan(coords_flat):
                    print('NaN in coords_flat after egnn')
                    print(coords_flat)
                    exit(-1 )

                # print(f'egnn : {egnn_layer_idx} finish')

                coords = coords_flat.view(B, L, 3)
                # 回填 h 供下一步融合
                fusion_out["hidden"] = h_flat.view(B, L, -1)

                x_in = fusion_out["hidden"]    # 供下一层 RFFL 使用


        fusion_out["hidden"], final_logits, final_prob = self.rffl.emb_layer_norm_after_lm_head(fusion_out["hidden"])

        if check_nan(coords):
            print('NaN in coords after emb_layer_norm_after_lm_head')
            exit(-1 )

        # print('fusion_out["hidden"] : ', fusion_out["hidden"].shape)

        # ---------- 4) 汇总输出 ----------
        return {
            "logits": final_logits,             # [B,L,V]
            "prob": final_prob,                 # [B,L,V]
            "hidden": fusion_out["hidden"],     # [B,L,E] 经过 EGNN 更新后的特征
            "coords": coords,                   # [B,L,3]
            "padding_mask": padding_mask,       # [B,L]
        }


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