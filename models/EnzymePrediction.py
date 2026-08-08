import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class EnzymePrediction(nn.Module):
    """
    残基类型预测头（单一分类头，不区分口袋/非口袋专用头）
    ---------------------------------------------------
    输入:
      - res_feat:    [B, L, E]   上游残基特征
      - pocket_mask: [B, L]      口袋布尔标记；不会被预测，仅可选用作条件
      - res_padding: [B, L]      padding 位置(True)将被屏蔽

    参数:
      - hidden_dim:  E
      - num_classes: 残基类别数（默认 21：20AA+UNK
      - dropout:     dropout 概率
      - use_pocket_as_feature: 是否把 pocket_mask 作为条件加入（FiLM 方式）

    输出:
      - logits: [B, L, C]
      - prob:   [B, L, C]
    """

    def __init__(self, cfg: dict, num_classes=None):
        super().__init__()
        hidden_dim = cfg.get('hidden_dim')
        self.num_classes = cfg.get('num_classes')
        dropout = cfg.get('dropout')
        self.use_pocket_as_feature = cfg.get('use_pocket_as_feature')
        self.device = torch.device(cfg.get('device', 'cpu'))

        self.norm = nn.LayerNorm(hidden_dim).to(self.device)
        self.drop = nn.Dropout(dropout).to(self.device)

        # 单一分类头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        ).to(self.device)

        # 可选：用 pocket 0/1 做 FiLM 条件
        if self.use_pocket_as_feature:
            self.film_gamma = nn.Linear(1, hidden_dim, bias=False).to(self.device)
            self.film_beta  = nn.Linear(1, hidden_dim, bias=True).to(self.device)

    def forward(
        self,
        res_feat: torch.Tensor,             # [B, L, E]
        pocket_mask: torch.Tensor = None,          # [B, L] (bool)
        res_padding: Optional[torch.Tensor] = None,  # [B, L] (bool)
    ) -> Dict[str, torch.Tensor]:

        # print(self.drop.device)
        # print(self.norm.device)
        # print(res_feat.device)


        x = self.drop(self.norm(res_feat))  # [B, L, E]

        # 仅作为条件特征，不做额外分类头
        if self.use_pocket_as_feature:
            cond = pocket_mask.to(x.dtype).unsqueeze(-1)  # [B, NoneL, 1]
            gamma = self.film_gamma(cond)                 # [B, L, E]
            beta  = self.film_beta(cond)                  # [B, L, E]
            x = (1.0 + gamma) * x + beta

        logits = self.head(x)                             # [B, L, C]

        if res_padding is not None:
            logits = logits.masked_fill(res_padding.unsqueeze(-1), -1e9)

        prob = F.softmax(logits, dim=-1)
        return {"logits": logits, "prob": prob}
