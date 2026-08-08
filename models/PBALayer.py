import sys
sys.path.append("..")
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PocketEnhancedBilevelAttentionLayer(nn.Module):
    """
    Pocket-enhanced Bilevel Attention
    ---------------------------------
    Inputs (per forward):
      - res_feat: FloatTensor [B, Lr, Er]      # NAELayer 输出的残基隐藏特征
      - res_coord: FloatTensor [B, Lr, 3]      # NAELayer 输出的残基坐标（Cα）
      - lig_feat: FloatTensor [B, Ll, El]      # SpatialNeighborhoodEquivariantLayer 的 h_out（配体原子特征）
      - lig_coord: FloatTensor [B, Ll, 3]      # SpatialNeighborhoodEquivariantLayer 的 x_out（配体原子坐标）
      - res_padding (可选): BoolTensor [B, Lr] # True=padding
      - lig_padding (可选): BoolTensor [B, Ll] # True=padding

    What it does:
      1) Residue↔Residue attention with RBF distance bias (PocketGen Eq.(3) 风格):
         a_rr = softmax_j( (Q_r K_r^T)/sqrt(d) + w_rr^T φ_rbf(||x_i - x_j||) )
      2) Residue↔Ligand-atom attention with RBF distance bias:
         a_rl(i,m) = softmax_m( (Q_rl K_l^T)/sqrt(d) + w_rl^T φ_rbf(||x_i - y_m||) )
         将未归一化分数 s_rl 先对配体原子做 logsumexp 聚合得到 s_i，再对残基做 softmax 得到 A_i。
      3) Pocket selection:
         pocket_mask = (A_i > pocket_threshold)
      4) Updates (参考 PocketGen Eq.(7)(8)(9) 思路):
         - Feature update（所有残基）:
             m_rr = a_rr @ V_r
             m_rl = a_rl @ V_l
             res_feat' = res_feat + W_o([m_rr || m_rl])
         - Coordinate update（仅口袋残基）:
             Δx_i = g_i * Σ_m a_rl(i,m) * (y_m - x_i)
             x_i' = x_i + 1[pocket] * Δx_i
           其中 g_i = tanh(W_g m_rl_i) 为标量门控
    Outputs:
      - res_feat_out: FloatTensor [B, Lr, Er]   # 更新后的残基特征
      - res_coord_out: FloatTensor [B, Lr, 3]   # 更新后的残基坐标（仅口袋位置被改动）
      - pocket_idx: List[Tensor] (len=B)        # 每个 batch 的口袋残基下标列表（可变长）
      - lig_feat_passthrough: FloatTensor [B, Ll, El] # 原样透传
      - lig_coord_passthrough: FloatTensor [B, Ll, 3] # 原样透传
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.device = cfg.get('device')
        self.res_dim = cfg.get('res_dim')
        self.lig_dim = cfg.get('lig_dim')
        self.d = cfg.get('attn_dim')
        self.rbf_k = cfg.get('rbf_k')
        self.rbf_cutoff = cfg.get('rbf_cutoff')
        self.pocket_threshold = cfg.get('pocket_threshold')
        self.allow_self_res_attn = cfg.get('allow_self_res_attn')
        self.s_gamma = cfg.get('s_gamma', 1.0)

        # --- Residue↔Residue projections ---
        self.q_r = nn.Linear(self.res_dim, self.d, bias=False)
        self.k_r = nn.Linear(self.res_dim, self.d, bias=False)
        self.v_r = nn.Linear(self.res_dim, self.d, bias=False)
        self.rbf_w_rr = nn.Linear(self.rbf_k, 1, bias=False)  # w_rr^T φ(d)

        # --- Residue↔Ligand projections ---
        self.q_rl = nn.Linear(self.res_dim, self.d, bias=False)
        self.k_l  = nn.Linear(self.lig_dim, self.d, bias=False)
        self.v_l  = nn.Linear(self.lig_dim, self.d, bias=False)
        self.rbf_w_rl = nn.Linear(self.rbf_k, 1, bias=False)

        # --- Output fusion (feature) ---
        self.out_proj = nn.Sequential(
            nn.Linear(self.d * 2, self.res_dim),
            nn.GELU(),
            nn.Linear(self.res_dim, self.res_dim),
        )

        # --- Coordinate gate (scalar in (−1,1) or unbounded) ---
        self.coord_gate = nn.Linear(self.d, 1)
        self.coord_gate_act = nn.Tanh() if cfg.get('coord_gate_tanh') else nn.Identity()

        # --- RBF centers & widths ---
        centers = torch.linspace(0.0, self.rbf_cutoff, self.rbf_k)
        # 相邻中心间距作为 σ 的基准；避免 0，给一个最小值
        sigma = (self.rbf_cutoff / (self.rbf_k - 1 + 1e-8))
        widths = torch.full((self.rbf_k,), sigma)
        self.register_buffer("rbf_centers", centers)  # [K]
        self.register_buffer("rbf_widths", widths)    # [K]

        self.to(self.device)


        # self.feat_gate = nn.Sequential(
        #     nn.Linear(self.d, 1),   # 从 m_rl 产生标量门控
        #     nn.Sigmoid(),           # β_f ∈ (0,1)
        # ).to(self.device)
        # self.pocket_proj = nn.Linear(self.d, self.res_dim).to(self.device)  # 将 m_rl 投影回残基特征维

    # ---------- utilities ----------

    # def _pairwise_dist2(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    #     """
    #     Squared pairwise Euclidean distance.
    #     X: [B, N, 3], Y: [B, M, 3] -> [B, N, M]
    #     """
    #     # (x - y)^2 = x^2 + y^2 - 2 x·y
    #     x2 = (X ** 2).sum(-1, keepdim=True)     # [B,N,1]
    #     y2 = (Y ** 2).sum(-1, keepdim=True).transpose(1, 2)  # [B,1,M]
    #     xy = X @ Y.transpose(1, 2)              # [B,N,M]
    #     d2 = x2 + y2 - 2.0 * xy                 # [B,N,M]
    #     return d2.clamp_min_(0.0)
    def _pairwise_dist2(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Squared pairwise Euclidean distance (数值稳定版).
        X: [B, N, 3], Y: [B, M, 3] -> [B, N, M]
        使用 float32 计算，避免 AMP/fp16 下 (X**2) 溢出。
        """
        # ===== 1) 在 float32 下计算，屏蔽 AMP =====
        # 注意：就这一小块禁用 autocast，对总算力影响很小
        with torch.cuda.amp.autocast(enabled=False):
            X32 = X.float()
            Y32 = Y.float()

            # 把潜在的 NaN/Inf 先清理掉（这里一般不会有，但多一层保险）
            X32 = torch.nan_to_num(X32, nan=0.0, posinf=0.0, neginf=0.0)
            Y32 = torch.nan_to_num(Y32, nan=0.0, posinf=0.0, neginf=0.0)

            # (x - y)^2 = x^2 + y^2 - 2 x·y
            x2 = (X32 ** 2).sum(-1, keepdim=True)                    # [B,N,1]
            y2 = (Y32 ** 2).sum(-1, keepdim=True).transpose(1, 2)    # [B,1,M]
            xy = X32 @ Y32.transpose(1, 2)                           # [B,N,M]

            d2 = x2 + y2 - 2.0 * xy                                  # [B,N,M]

            # ===== 2) 数值安全处理：去掉 NaN/Inf，限制范围 =====
            # 先把 NaN/Inf 变成有限值
            d2 = torch.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)

            # 理论上 d2 >= 0，负数只可能来自数值误差
            d2.clamp_(min=0.0)

            # 再做一个上界裁剪，防止极端大距离影响后续 RBF
            # 这里的上界可以和你 PBA 里用的 max_dist 对齐
            max_dist = getattr(self, "max_dist", 200.0)   # 单位 Å，物理上已经很宽松
            max_d2   = max_dist * max_dist                # 默认 4e4
            d2.clamp_(max=max_d2)

        # 返回到原始 dtype（fp16/fp32）
        return d2.to(X.dtype)

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
        """
        RBF expansion for distances d (Å).
        d: [*] -> out: [*, K]
        φ_k(d) = exp( - (d - c_k)^2 / (2 σ_k^2) )
        """
        B = d.shape
        d = d.unsqueeze(-1)                           # [..., 1]
        c = self.rbf_centers.view(*([1] * len(B)), -1)  # broadcast
        s = self.rbf_widths.view(*([1] * len(B)), -1)
        z = (d - c) / (s + 1e-8)
        return torch.exp(-0.5 * z * z)               # [..., K]

    def _apply_mask_logits(self, logits: torch.Tensor, mask_j: Optional[torch.Tensor], fill_value: float = -1e9):
        """
        logits: [B, N, M], mask_j: [B, M] True=padding → 被屏蔽
        """
        if mask_j is None:
            return logits
        mask = mask_j.unsqueeze(1)                   # [B,1,M]
        return logits.masked_fill(mask, fill_value)

    # ---------- forward ----------

    def forward(
        self,
        res_feat: torch.Tensor,          # [B, Lr, Er]
        res_coord: torch.Tensor,         # [B, Lr, 3]
        lig_feat: torch.Tensor,          # [B, Ll, El]
        lig_coord: torch.Tensor,         # [B, Ll, 3]
        res_padding: Optional[torch.Tensor] = None, # [B, Lr] True=pad
        lig_padding: Optional[torch.Tensor] = None, # [B, Ll] True=pad
    ):
        """
        OUT:
        res_feat_out
        res_coord_out
        pocket_idx
        pocket_mask
        lig_feat
        lig_coord
        """


        if _check_finite(res_feat, "res_feat") or \
           _check_finite(res_coord, "res_coord") or \
           _check_finite(lig_feat, "lig_feat") or \
              _check_finite(lig_coord, "lig_coord"):
            exit(-1)

        # print('res_feat', res_feat.shape)
        device = self.device
        B, Lr, Er = res_feat.shape
        _, Ll, El = lig_feat.shape

        # sanitize coords
        res_coord = torch.where(torch.isfinite(res_coord), res_coord, torch.zeros_like(res_coord))
        lig_coord = torch.where(torch.isfinite(lig_coord), lig_coord, torch.zeros_like(lig_coord))

        # ========= 1) Residue-Residue attention with RBF bias =========
        q_r = self.q_r(res_feat)                           # [B,Lr,d]
        k_r = self.k_r(res_feat)                           # [B,Lr,d]
        v_r = self.v_r(res_feat)                           # [B,Lr,d]

        attn_rr = torch.matmul(q_r, k_r.transpose(1, 2))   # [B,Lr,Lr]
        attn_rr = attn_rr / math.sqrt(self.d)

        # distance bias
        d2_rr = self._pairwise_dist2(res_coord, res_coord) # [B,Lr,Lr]
        d_rr = torch.sqrt(d2_rr + 1e-8)
        rbf_rr = self._rbf(d_rr)                           # [B,Lr,Lr,K]
        bias_rr = self.rbf_w_rr(rbf_rr).squeeze(-1)        # [B,Lr,Lr]
        # print('attn_rr', attn_rr.shape)
        # print('bias_rr', bias_rr.shape)
        logits_rr = attn_rr + bias_rr

        _check_finite(d2_rr, 'd2_rr')
        _check_finite(d_rr, 'd_rr')
        _check_finite(rbf_rr, 'rbf_rr')
        _check_finite(bias_rr, 'bias_rr')

        if res_padding is not None:
            logits_rr = self._apply_mask_logits(logits_rr, res_padding, fill_value=-1e9)

        if not self.allow_self_res_attn:
            eye = torch.eye(Lr, device=device).bool()
            logits_rr = logits_rr.masked_fill(eye.unsqueeze(0), -1e9)

        a_rr = F.softmax(logits_rr, dim=-1)                # softmax over j
        m_rr = torch.matmul(a_rr, v_r)                      # [B,Lr,d]

        # ========= 2) Residue↔Ligand-atom attention with RBF bias =========
        q_rl = self.q_rl(res_feat)                          # [B,Lr,d]
        k_l  = self.k_l(lig_feat)                           # [B,Ll,d]
        v_l  = self.v_l(lig_feat)                           # [B,Ll,d]

        attn_rl = torch.matmul(q_rl, k_l.transpose(1, 2))   # [B,Lr,Ll]
        attn_rl = attn_rl / math.sqrt(self.d)

        d2_rl = self._pairwise_dist2(res_coord, lig_coord)  # [B,Lr,Ll]
        d_rl = torch.sqrt(d2_rl + 1e-8)
        rbf_rl = self._rbf(d_rl)                            # [B,Lr,Ll,K]
        bias_rl = self.rbf_w_rl(rbf_rl).squeeze(-1)         # [B,Lr,Ll]
        logits_rl = attn_rl + bias_rl

        _check_finite(d2_rl, 'd2_rl')
        _check_finite(d_rl, 'd_rl')
        _check_finite(rbf_rl, 'rbf_rl')
        _check_finite(bias_rl, 'bias_rl')



        if lig_padding is not None:
            logits_rl = self._apply_mask_logits(logits_rl, lig_padding, fill_value=-1e9)

        # 注意力（对配体原子归一化）
        a_rl = F.softmax(logits_rl, dim=-1)                 # [B,Lr,Ll]
        m_rl = torch.matmul(a_rl, v_l)                      # [B,Lr,d]

        # 残基-配体注意力分数 A_i：
        # 先按原子维做 logsumexp 聚合，再在残基维做 softmax → A_i
        s_i = torch.logsumexp(logits_rl, dim=-1)            # [B,Lr]
        A = F.softmax(s_i, dim=-1)                          # [B,Lr] 归一化到残基

        # ========= 3) Feature update（所有残基+口袋残基） =========
        feat_cat = torch.cat([m_rr, m_rl], dim=-1)          # [B,Lr, 2d]
        delta_feat = self.out_proj(feat_cat)                # [B,Lr, Er]
        # beta_f       = self.feat_gate(m_rl)                 # [B,Lr,1] 标量门控（不随旋转改变）
        # delta_pocket = self.pocket_proj(beta_f * m_rl)      # [B,Lr,Er]
        # delta_pocket = delta_pocket * pocket_mask.unsqueeze(-1).float()

        # res_feat_out = res_feat + delta_feat + delta_pocket

        mu  = s_i.mean(dim=1, keepdim=True)
        sig = s_i.std(dim=1, keepdim=True).clamp_min(1e-6)
        s_n = (s_i - mu) / sig                                                      #  转为标准分布
        gamma = getattr(self, "s_gamma", 1.0)
        s_i_g = torch.sigmoid(gamma * s_n.float()).unsqueeze(-1).to(delta_feat.dtype)   # [B, Lr, 1]

        # ========= 3.1) Pocket selection via threshold on s_i_g =========
        pocket_mask = (s_i_g.squeeze(-1) > self.pocket_threshold)           # [B,Lr] bool
        # 组装 pocket 下标（Python 列表，方便可变长）
        pocket_idx: List[torch.Tensor] = []
        for b in range(B):
            idx = torch.nonzero(pocket_mask[b], as_tuple=False).flatten()
            pocket_idx.append(idx)

        delta_feat   = delta_feat * s_i_g
        # delta_pocket = delta_pocket * s_i_g
        # delta_pocket = delta_pocket * pocket_mask.unsqueeze(-1).float()

        res_feat_out = res_feat + delta_feat                                    # + delta_pocket


        # ========= 5) Coordinate update（仅口袋残基） =========
        # Δx_i = g_i * Σ_m a_rl(i,m) (y_m - x_i)
        # g_i: [B,Lr,1]
        g = self.coord_gate_act(self.coord_gate(m_rl))      # [B,Lr,1]
        # (y_m - x_i) 广播： [B,Lr,Ll,3]

        # —— 残基→配体位移（RL）——
        vec_il  = lig_coord.unsqueeze(1) - res_coord.unsqueeze(2)               # [B,Lr,Ll,3]
        dist_il = vec_il.norm(dim=-1, keepdim=True).clamp_min(1e-6)            # [B,Lr,Ll,1]
        dir_il  = vec_il / dist_il                                             # [B,Lr,Ll,3]  单位方向

        lig_valid = (~lig_padding).unsqueeze(1).unsqueeze(-1).to(dir_il.dtype)

        delta_rl = (a_rl.unsqueeze(-1) * dir_il * lig_valid).sum(dim=2)             # [B,Lr,3]

        # —— 残基→残基位移（RR）——
        # (x_j - x_i)： [B,Lr,Lr,3]
        vec_ij  = res_coord.unsqueeze(2) - res_coord.unsqueeze(1)                   # [B,Lr,Lr,3]
        dist_ij = vec_ij.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_ij  = vec_ij / dist_ij                                                  # [B,Lr,Lr,3]

        # 去自环
        I = torch.eye(res_coord.size(1), device=res_coord.device, dtype=dir_ij.dtype)  # [Lr,Lr]
        I = I.view(1, *I.shape, 1)                                                     # [1,Lr,Lr,1]
        dir_ij = dir_ij * (1.0 - I)                                                    # 置零对角方向

        j_valid = (~res_padding).unsqueeze(1).unsqueeze(-1).to(dir_ij.dtype)

        delta_rr = (a_rr.unsqueeze(-1) * dir_ij * j_valid).sum(dim=2)               # [B,Lr,3]

        delta_x = (delta_rl + delta_rr)                                             # [B,Lr,3]
        delta_x = g * s_i_g * delta_x                                              # [B,Lr,3]


        res_coord_out = res_coord + delta_x


        # 仅口袋位置更新
        # delta_x = (a_rl.unsqueeze(-1) * dir_il).sum(dim=2)                     # [B,Lr,3]
        # delta_x = g * delta_x



        if _check_finite(res_feat_out, "res_feat_out") or \
           _check_finite(res_coord_out, "res_coord_out") or \
           _check_finite(lig_feat, "lig_feat") or \
              _check_finite(lig_coord, "lig_coord"):
            exit(-1)



        #（可选）padding 位置强制还原
        if res_padding is not None:
            res_coord_out = torch.where(res_padding.unsqueeze(-1), res_coord, res_coord_out)
            res_feat_out  = torch.where(res_padding.unsqueeze(-1), res_feat,  res_feat_out)

        return res_feat_out, res_coord_out, pocket_idx, pocket_mask, lig_feat, lig_coord, logits_rl, s_i_g



def _check_finite(x, name):
    if not torch.isfinite(x).all():
        print(f"[PBA] non-finite in {name}:",
              "nan=", torch.isnan(x).sum().item(),
              "posinf=", torch.isinf(x).sum().item(),
              "neginf=", torch.isinf(-x).sum().item())
        return True
    return False
