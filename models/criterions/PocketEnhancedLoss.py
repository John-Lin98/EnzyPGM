# criterion.py
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
batch_num = 0


def _to_bool(t: torch.Tensor) -> torch.Tensor:
    return t.bool() if t.dtype != torch.bool else t


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [...], mask: same shape except last dim (or broadcastable); returns mean over mask==False."""
    mask = mask.to(dtype=x.dtype)
    num = (x * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def _build_pocket_gt_mask(
    pocket_idxs: Union[List[List[int]], torch.Tensor],
    B: int,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """
    pocket_idxs: list of length B, each a list[int] for that sample's pocket indices,
                 or a padded LongTensor [B, Lp] with -1 for pad.
    Returns: BoolTensor [B, L] (True at GT pocket positions).
    """

    if pocket_idxs is None:
        return None

    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    if isinstance(pocket_idxs, list):
        for b, idxs in enumerate(pocket_idxs):
            if len(idxs) > 0:
                idx = torch.tensor(idxs, device=device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < L)]
                if idx.numel() > 0:
                    mask[b, idx] = True
    else:
        # assume tensor [B, Lp], with -1 for padding
        valid = pocket_idxs.ge(0)
        for b in range(B):
            idx = pocket_idxs[b][valid[b]]
            idx = idx[(idx >= 0) & (idx < L)]
            if idx.numel() > 0:
                mask[b, idx] = True
    return mask


class PocketEnhancedLoss(torch.nn.Module):
    """
    计算训练时的 loss（四部分）：
      1) 掩码位置的序列交叉熵（Masked LM）
      2) 掩码位置的坐标 SmoothL1
      3) 口袋位置的序列交叉熵（口袋全体 or 与 mask 交集）
      4) 口袋位置的坐标 SmoothL1

    cfg 可配置项（示例/默认）：
      cfg = {
        "weights": {
          "lm": 1.0,
          "coord": 1.0,
          "pocket_coord": 1.0,
          "pocket_gate": 1.0
        },
        "pocket_target": "all",        # "all" 或 "mask_intersect"
        "ignore_index": -100,          # CE 的 ignore_index
        "smoothl1_beta": 1.0,          # SmoothL1 的 beta（Huber delta）
        "coord_scale": 1.0,            # 坐标差的缩放（单位不一致时可调）
        "pocket_gate_l2": 0.0,         # 口袋 gate 的 L2 正则系数
        "pocket_ratio": null           # 先验口袋比例，用于 BCE 加权
      }
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg or {}
        w = self.cfg.get("weights", {})
        self.w_lm = float(w.get("lm", 1.0))
        self.w_coord = float(w.get("coord", 1.0))
        self.w_p_coord = float(w.get("pocket_coord", 1.0))
        self.pocket_gate_l2 = float(w.get("pocket_gate_l2", 0.0))
        # dedicated weight for pocket gate BCE (fallback to legacy pocket_lm if present)
        self.w_p_g = float(w.get("pocket_gate", w.get("pocket_lm", 1.0)))

        self.pocket_target = str(self.cfg.get("pocket_target", "all")).lower()
        assert self.pocket_target in ("all", "mask_intersect")

        self.ignore_index = int(self.cfg.get("ignore_index", -100))
        self.smoothl1_beta = float(self.cfg.get("smoothl1_beta", 1.0))
        self.coord_scale = float(self.cfg.get("coord_scale", 1.0))

        pocket_ratio = self.cfg.get("pocket_ratio", None)
        if pocket_ratio is not None:
            pocket_ratio = float(pocket_ratio)
            eps = 1e-6
            self.pocket_ratio = min(max(pocket_ratio, eps), 1 - eps)
        else:
            self.pocket_ratio = None

    @torch.no_grad()
    def _make_labels_with_ignore(
        self,
        targets: torch.Tensor,         # [B,L]
        select_mask: torch.Tensor,     # [B,L] True=use this position
        ignore_index: int,
    ) -> torch.Tensor:
        """
        Build label tensor for F.cross_entropy with ignore_index elsewhere.
        """
        labels = torch.full_like(targets, fill_value=ignore_index)
        labels[select_mask] = targets[select_mask]
        return labels

    def forward(
        self,
        pegm_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        pegm_out keys (from PEGM forward):
          - enzyme_pred: dict with 'logits' [B,L,V] (and maybe 'prob')
          - res_feats:   [B,L,E]  (unused by losses here)
          - res_coords:  [B,L,3]  (pred coords)
          - pocket_idx:  anything (not used here)
          - pocket_mask: [B,L]    (pred pocket mask; not used for GT losses)
          - lig_feats, lig_coords, res_padding, lig_padding, mask
        batch keys (from init_data):
          - src_tokens:  [B,L]    (GT sequence ids)
          - coords:      [B,L,3]  (GT coords)
          - pocket_idxs: list[list[int]] or padded tensor with -1
          - res_padding: [B,L]    True=pad
          - mask:        [B,L]    1=masked (the MLM mask used by model)
        """
        device = pegm_out["res_coords"].device

        # ====== Common tensors ======
        logits = pegm_out["enzyme_pred"]["logits"]             # [B,L,V]
        pred_coords = pegm_out["res_coords"]                   # [B,L,3]
        res_padding = _to_bool(pegm_out.get("res_padding", batch["res_padding"]))  # [B,L]
        mask = pegm_out.get("mask", batch["mask"]).to(device)  # [B,L], 1=masked
        s_i_g = pegm_out.get("s_i_g", None)

        pocket_idx = batch["pocket_idxs"]
        gt_tokens = batch["src_tokens_wo_mask"].to(device)             # [B,L]
        gt_coords = batch["coords_wo_mask"].to(device)                 # [B,L,3]

        # print('gt_tokens', gt_tokens)

        B, L = gt_tokens.shape
        V = logits.size(-1)
        assert logits.shape[:2] == (B, L), "logits and targets length mismatch"
        assert pred_coords.shape[:2] == (B, L), "coords length mismatch"

        not_pad = ~res_padding.bool()                          # True=valid residue

        # ========= 1) Masked LM loss（只在 mask==1 & 非 padding） =========
        mlm_sel = (mask == True) & not_pad                        # [B,L]
        if mlm_sel.any():
            labels_mlm = self._make_labels_with_ignore(gt_tokens, mlm_sel, self.ignore_index)  # [B,L]
            valid_mlm = labels_mlm.ne(self.ignore_index)
            labels_mlm[valid_mlm] -= 4  # <—— 减去偏移量
            # print(labels_mlm)
            loss_lm = F.cross_entropy(
                logits.view(-1, V),
                labels_mlm.view(-1),
                ignore_index=self.ignore_index,
                reduction="mean",
            )
        else:
            loss_lm = torch.tensor(0.0, device=device)

        # print("logits", logits.shape, logits.dtype, logits.device, "targets", labels_mlm.shape, labels_mlm.dtype, labels_mlm.device)
        # if 'mlm_sel' in locals():
        #     print('mlm_sel.dtype', mlm_sel.dtype)
        #     print('mlm_sel.shape', mlm_sel.shape)
        #     print('mlm_sel.device', mlm_sel.device)

        # C = logits.size(-1)
        # ignore = self.ignore_index
        # t = labels_mlm

        # assert t.dtype == torch.long
        # if ignore is None:
        #     assert (t.min() >= 0) and (t.max() < C), f"min={t.min()} max={t.max()} C={C}"
        # else:
        #     bad = ~((t >= 0) & (t < C) | (t == ignore))
        #     print("bad count:", bad.sum().item())
        #     assert not bad.any()

        # #
        # for name, x in [("logits", logits), ("targets", t)]:
        #     assert torch.isfinite(x).all(), f"{name} has NaN/Inf"

        # print('pred_coords : ', pred_coords.shape)
        # print('gt_coords : ', gt_coords.shape)
        # ========= 2) 掩码位置坐标 SmoothL1 =========
        # per-node loss: mean over last dim (3D)，再按掩码平均
        if mlm_sel.any():
            coord_diff = (pred_coords - gt_coords) * self.coord_scale  # [B,L,3]
            loss_coord_all = F.smooth_l1_loss(
                pred_coords, gt_coords, beta=self.smoothl1_beta, reduction="none"
            ).mean(dim=-1)  # [B,L]
            loss_coord = _masked_mean(loss_coord_all, mlm_sel)
        else:
            loss_coord = torch.tensor(0.0, device=device)

        global batch_num
        batch_num += 1
        # if loss_coord.detach().item() > 20.0 or loss_coord.detach().item() < 1.0:
        #     print('true num : ', ((mlm_sel).to(dtype=pred_coords.dtype)).sum())
        #     print('all num : ', ((mlm_sel).to(dtype=pred_coords.dtype)).sum()+((~mlm_sel).to(dtype=pred_coords.dtype)).sum())
        #     print('pred nonzero num : ', torch.count_nonzero(pred_coords) / 3)
        #     print('gt nonzero num : ', torch.count_nonzero(gt_coords) / 3)
        #     # print('mask : ', mask)

        #     print('pred_coords', pred_coords[mlm_sel])
        #     print('gt_coords', gt_coords[mlm_sel])
        #     print('loss_coord', loss_coord)
            # exit(-1)


        # ========= 3) 口袋位置序列 CE =========
        pocket_gt_mask = _build_pocket_gt_mask(
            batch["pocket_idxs"], B=B, L=L, device=device
        )                                                     # [B,L] True at GT pockets

        if pocket_gt_mask is not None:
            if self.pocket_target == "mask_intersect":
                pocket_sel = pocket_gt_mask & mlm_sel             # 与 MLM 掩码取交集
            else:
                pocket_sel = pocket_gt_mask & not_pad             # 口袋全体（排除 padding）

            if pocket_sel.any():
                labels_p = self._make_labels_with_ignore(gt_tokens, pocket_sel, self.ignore_index)
                valid_p = labels_p.ne(self.ignore_index)
                labels_p[valid_p] -= 4  # <—— 减偏移
                loss_p_lm = F.cross_entropy(
                    logits.view(-1, V),
                    labels_p.view(-1),
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )
            else:
                loss_p_lm = torch.tensor(0.0, device=device)

            # ========= 4) 口袋位置坐标 SmoothL1 =========
            if pocket_sel.any():
                loss_p_coord_all = F.smooth_l1_loss(
                    pred_coords, gt_coords, beta=self.smoothl1_beta, reduction="none"
                ).mean(dim=-1)  # [B,L]
                loss_p_coord = _masked_mean(loss_p_coord_all, pocket_sel)
            else:
                loss_p_coord = torch.tensor(0.0, device=device)
        else :
            loss_p_lm = torch.tensor(0.0, device=device)
            loss_p_coord = torch.tensor(0.0, device=device)

        # ========== 5）口袋位置预测loss ==========
        loss_pocket_gate_reg = torch.tensor(0.0, device=device)
        if s_i_g is not None:
            B, Lr, _ = s_i_g.shape
            device   = s_i_g.device
            dtype    = s_i_g.dtype
            tgt = torch.zeros(B, Lr, 1, device=device, dtype=dtype)  # [B, Lr, 1]
            for b, idxs in enumerate(batch["pocket_idxs"]):
                if idxs:  # 可能为空
                    idxs_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
                    # 可选：保护越界
                    # idxs_t = idxs_t[(idxs_t >= 0) & (idxs_t < Lr)]
                    tgt[b, idxs_t, 0] = 1.0

            valid = (~batch["res_padding"]).to(device)          # [B, Lr]

            valid = valid.unsqueeze(-1)                              # [B, Lr, 1]

            #类别不均衡加权（pos_weight 自适应）
            with torch.no_grad():
                if self.pocket_ratio is not None:
                    pos_ratio = torch.tensor(self.pocket_ratio, device=device, dtype=dtype)
                    pos_w = 1.0 / pos_ratio
                    neg_w = 1.0 / (1.0 - pos_ratio)
                else:
                    pos = (tgt * valid).sum().clamp_min(1.0)        # 正类数
                    tot = valid.sum().clamp_min(1.0)                # 有效位总数
                    neg = tot - pos
                    pos_w = (neg / pos).to(dtype)                   # 正类增权系数
                    neg_w = torch.tensor(1.0, device=device, dtype=dtype)

                # 为每个位置构造权重：正类用 pos_w，负类用 neg_w
                w = torch.where(tgt > 0, pos_w, neg_w).to(dtype)

            p = s_i_g.clamp(1e-6, 1 - 1e-6)                         # [B, Lr, 1]
            bce_map = F.binary_cross_entropy(p, tgt, reduction="none")  # [B, Lr, 1]
            num = valid.to(dtype)
            loss_pocket_gate = (bce_map * w * num).sum() / num.sum().clamp_min(1.0)
            if self.pocket_gate_l2 != 0.0:
                sq = (s_i_g ** 2) * num
                loss_pocket_gate_reg = sq.sum() / num.sum().clamp_min(1.0)
        else:
            loss_pocket_gate = torch.tensor(0.0, device=device)

        # ========= 汇总 =========
        total = (
            self.w_lm * loss_lm
            + self.w_coord * loss_coord
            + self.w_p_coord * loss_p_coord
            + self.w_p_g * loss_pocket_gate
            + self.pocket_gate_l2 * loss_pocket_gate_reg
        )

        # if loss_coord.requires_grad == True:
        #     print("loss_coord.requires_grad == True !!!!! ")

        return {
            "loss": total,
            "loss_lm": loss_lm.detach(),
            "loss_coord": loss_coord.detach(),
            "loss_pocket_lm": loss_p_lm.detach(),
            "loss_pocket_coord": loss_p_coord.detach(),
            "loss_pocket_gate": loss_pocket_gate.detach(),
            "loss_pocket_gate_reg": loss_pocket_gate_reg.detach(),
        }
