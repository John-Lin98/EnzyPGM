import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class PocketSubstrateBindingPrediction(nn.Module):
    """
    口袋-底物结合强度预测头（简单版）
    -------------------------------------------------------
    输入（来自两个上游模块）：
      - h_out         : [B, Ll, Hl]   （SNELayer 原子级特征，Ll 为配体原子数）
      - x_out         : [B, Ll, 3]    （SNELayer 原子级坐标）
      - res_feat_out  : [B, Lr, Hr]   （PBALayer 残基特征）
      - res_coord_out : [B, Lr, 3]    （PBALayer 残基坐标）
      - logits_rl     : [B, Lr, Ll]   （PBALayer 残基-配体原子注意力的“未归一化打分”）

    输出：
      - pred: [B, 3]  分别为 (HP_count, HB_count, PiPi_count)，用 softplus 保证非负实数
                      如需整数，可在外部做 round/clamp。

    说明：
      - 模型思路非常“轻量”：
        1) 由 logits_rl 计算残基级权重 A_i 以及残基→配体原子注意力 a_rl(i,m)；
        2) 以 A_i 聚合口袋特征、以 a_rl 聚合配体消息（到残基），再以 A_i 聚合成全局交互特征；
        3) 计算少量几何统计（加权口袋中心、配体中心、两者距离/最近距离）；
        4) 将 (口袋聚合特征, 配体聚合特征, 交互特征, 几何统计) 拼接，经小 MLP 输出 3 维强度。
    """

    def __init__(
        self,
        res_dim: int,               # Hr  残基特征维度
        lig_dim: int,               # Hl  配体原子特征维度
        proj_dim: int = 256,        # 投影到公共维度
        hidden_mlp: int = 256,      # MLP 隐藏维
        dropout: float = 0.1,
    ):
        super().__init__()
        self.res_norm = nn.LayerNorm(res_dim)
        self.lig_norm = nn.LayerNorm(lig_dim)

        # 统一投影到公共维度
        self.res_proj = nn.Sequential(
            nn.Linear(res_dim, proj_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, proj_dim), nn.GELU(), nn.Dropout(dropout)
        )

        # 交互消息再精炼一下
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout)
        )

        # 最终回归头：输入 = 口袋聚合 + 配体聚合 + 交互聚合 + 几何统计(4维)
        self.out_head = nn.Sequential(
            nn.Linear(proj_dim * 3 + 4, hidden_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, 3),
        )

        # 非负输出
        self.nonneg = nn.Softplus(beta=1.0)

    @staticmethod
    def _safe_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
        # 数值稳定 softmax
        return F.softmax(x - x.max(dim=dim, keepdim=True).values, dim=dim)

    def forward(
        self,
        h_out: torch.Tensor,                 # [B, Ll, Hl]
        x_out: torch.Tensor,                 # [B, Ll, 3]
        res_feat_out: torch.Tensor,          # [B, Lr, Hr]
        res_coord_out: torch.Tensor,         # [B, Lr, 3]
        logits_rl: torch.Tensor,             # [B, Lr, Ll]
        res_padding: Optional[torch.Tensor] = None,  # [B, Lr] True=pad（可选）
        lig_padding: Optional[torch.Tensor] = None,  # [B, Ll] True=pad（可选）
    ) -> Dict[str, torch.Tensor]:
        B, Lr, Hr = res_feat_out.shape
        _, Ll, Hl = h_out.shape

        # ---- 1) 归一化 + 投影到公共维度 ----
        res_feat = self.res_proj(self.res_norm(res_feat_out))   # [B, Lr, P]
        lig_feat = self.lig_proj(self.lig_norm(h_out))          # [B, Ll, P]

        # ---- 2) 由 logits_rl 推出注意力 ----
        # a_rl: 对每个残基 i，沿配体原子维 softmax
        a_rl = self._safe_softmax(logits_rl, dim=-1)            # [B, Lr, Ll]

        # A_i：残基级权重（logsumexp 聚合原子后，对残基 softmax）
        s_i = torch.logsumexp(logits_rl, dim=-1)                # [B, Lr]
        A = self._safe_softmax(s_i, dim=-1)                     # [B, Lr]

        # 可选：屏蔽 padding 对注意力的影响
        if res_padding is not None:
            A = A * (~res_padding).float()
            A = A / (A.sum(dim=-1, keepdim=True).clamp_min(1e-6))  # 重新归一化
        if lig_padding is not None:
            a_rl = a_rl * (~lig_padding).float().unsqueeze(1)       # [B,Lr,Ll]
            denom = a_rl.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            a_rl = a_rl / denom

        # ---- 3) 口袋/配体聚合特征 ----
        # 口袋聚合：按 A_i 聚合残基特征
        pocket_pool = torch.sum(res_feat * A.unsqueeze(-1), dim=1)      # [B, P]
        # 配体聚合：平均（或按 a_rl 的总分布加权，这里简化为平均）
        if lig_padding is None:
            lig_pool = lig_feat.mean(dim=1)                              # [B, P]
        else:
            valid = (~lig_padding).float().unsqueeze(-1)                 # [B,Ll,1]
            lig_pool = (lig_feat * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        # 交互聚合：先将配体消息聚到每个残基，再按 A_i 聚合到全局
        # m_rl(i) = Σ_m a_rl(i,m) * lig_feat(m)
        m_rl = torch.matmul(a_rl, lig_feat)                               # [B, Lr, P]
        inter_pool = torch.sum(self.inter_proj(m_rl) * A.unsqueeze(-1), dim=1)  # [B, P]

        # ---- 4) 简单几何统计 ----
        # 口袋中心（按 A_i 加权）、配体中心（简单平均）
        pocket_center = torch.sum(res_coord_out * A.unsqueeze(-1), dim=1)     # [B, 3]
        if lig_padding is None:
            lig_center = x_out.mean(dim=1)                                   # [B, 3]
        else:
            valid = (~lig_padding).float().unsqueeze(-1)                      # [B,Ll,1]
            lig_center = (x_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        # 口袋中心与配体中心的欧氏距离
        center_dist = torch.norm(lig_center - pocket_center, dim=-1, keepdim=True)  # [B,1]

        # 最近距离（残基↔配体原子）：min_i,m ||x_i - y_m||
        # 简化为：按 A_i 加权的残基中心到每个原子的最小距离（数值更稳）
        diff = x_out.unsqueeze(1) - pocket_center.unsqueeze(1)          # [B,1,Ll,3]
        lig_dists = torch.norm(diff, dim=-1)                            # [B,1,Ll]
        min_lig_dist = lig_dists.min(dim=-1, keepdim=True).values.squeeze(1)  # [B,1]

        # 配体原子数（归一化尺度，避免大分子偏置），以及 A 的熵（口袋稀疏度）
        lig_len = (Ll if lig_padding is None else (~lig_padding).float().sum(dim=-1, keepdim=True))  # [B,1]
        lig_len_log = torch.log(lig_len.clamp_min(1.0))                                             # [B,1]
        A_safe = (A + 1e-8)
        A_entropy = -(A_safe * A_safe.log()).sum(dim=-1, keepdim=True) / A_safe.size(1)             # [B,1]

        geom_feat = torch.cat([center_dist, min_lig_dist, lig_len_log, A_entropy], dim=-1)          # [B,4]

        # ---- 5) 拼接并预测 ----
        global_feat = torch.cat([pocket_pool, lig_pool, inter_pool], dim=-1)  # [B, 3P]
        fused = torch.cat([global_feat, geom_feat], dim=-1)                   # [B, 3P+4]
        raw_out = self.out_head(fused)                                        # [B, 3]
        pred = self.nonneg(raw_out)                                           # [B, 3] 非负

        return {"pred": pred}  # 对应 (HP_count, HB_count, PiPi_count)
