#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEGM inference on validation set
--------------------------------
- 加载 config + ckpt
- 在 valid_data_path 上跑一遍模型
- 输出 JSONL，每行包含 ground_truth / prediction 两部分
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
from torch import nn

from models.PEGM import PocketAugmentedEnzymeGenerativeModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ====== 一个简单的 JSONL → batch 迭代器（按 max_tokens 打包） ======
def jsonl_batch_iterator(
    jsonl_path: str,
    max_tokens: int = 1024,
) -> List[List[Dict[str, Any]]]:
    """
    按“蛋白序列长度总和 ≤ max_tokens”打包成 batch
    每个 batch 是 List[dict]，可以直接喂入 model.init_data(...)
    """
    batch = []
    cur_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # 序列可能是字符串或 [str]，与你 init_data 里一致
            seq = sample["seqs"][0] if isinstance(sample["seqs"], (list, tuple)) else sample["seqs"]
            L = len(seq)

            # 如果再加这个 sample 会超过 max_tokens，就先把当前 batch yield 掉
            if batch and cur_tokens + L > max_tokens:
                yield batch
                batch = []
                cur_tokens = 0

            batch.append(sample)
            cur_tokens += L

    if batch:
        yield batch


def jsonl_single_iterator(jsonl_path: str) -> List[List[Dict[str, Any]]]:
    """逐样本 yield，batch size 恒为 1。"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            yield [sample]


def load_model(cfg_model: Dict[str, Any], ckpt_path: str, device: torch.device) -> nn.Module:
    """
    1) 构造 PEGM 模型
    2) 加载 ckpt 权重（兼容 'model' / 直接 state_dict 两种）
    """
    model = PocketAugmentedEnzymeGenerativeModel(cfg_model).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  # 如果训练时 torch.save({"model": model.state_dict(), ...})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys in state_dict:", missing)
    if unexpected:
        print("[WARN] Unexpected keys in state_dict:", unexpected)

    return model


def tensor_to_list(t: torch.Tensor):
    """方便转成 Python 原生 list（先搬到 cpu 再 .tolist）"""
    return t.detach().cpu().tolist()

def load_substrate_map(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    读取 seq/pdb -> substrate 的映射，文件为 JSONL，每行包含字段:
      - seq (可选)
      - pdb 或 pdbs (可选，如 "4k4s.A")
      - substrate (必选)
    只取首次出现，后续重复不覆盖。
    """
    seq_map: Dict[str, str] = {}
    pdb_map: Dict[str, str] = {}
    print('load substrate map from : ', path)
    # exit()
    if not path:
        return seq_map, pdb_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = obj.get("seq")
            pdb = obj.get("pdb")
            # print('pdb in load substrate map : ', pdb)
            pdbs = obj.get("pdbs")
            substrate = obj.get("substrate")
            # print('substrate in load substrate map : ', substrate)
            if isinstance(seq, str) and substrate is not None and seq not in seq_map:
                seq_map[seq] = substrate
            if substrate is not None:
                # 单个 pdb
                if isinstance(pdb, str):
                    key = pdb.strip().lower()
                    if key and key not in pdb_map:
                        pdb_map[key] = substrate
                # pdbs 列表
                if isinstance(pdbs, (list, tuple)):
                    for item in pdbs:
                        if isinstance(item, str):
                            key = item.strip().lower()
                            if key and key not in pdb_map:
                                pdb_map[key] = substrate
    return seq_map, pdb_map

def normalize_pdb_id(val: Any) -> str:
    """规范 pdb 标识，取字符串形式并小写。"""
    return str(val).strip().lower()

def normalize_pdb_id(val: Any) -> str:
    """规范 pdb 标识，取字符串形式并小写。"""
    return str(val).strip().lower()

def normalize_ec4(val: Any) -> str:
    """将任意 ec4 表示规范为 'a.b.c.d'，不足补 0。"""
    digits = re.findall(r"\d+", str(val))
    while len(digits) < 4:
        digits.append("0")
    return ".".join(digits[:4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="训练时使用的 config JSON 路径")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="待加载的模型权重路径（.pt/.pth 等）")
    parser.add_argument("--out", type=str, default="",
                        help="输出目录（默认：和 ckpt 同目录，自动创建子目录）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="打包 batch 的 token 上限（默认使用 config 里的 max_tokens）")
    parser.add_argument("--gen_batch_size", type=int, default=None,
                        help="可选：生成阶段强制 batch size（设为 1 可逐样本推理），不影响训练")
    parser.add_argument("--substrate_db", type=str, default=None,
                        help="可选：全量 JSONL，包含字段 seq 和 substrate，用于补全 ground_truth.substrate")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 加载 config
    cfg = load_config(args.config)
    cfg_model = cfg["model"]
    cfg_train = cfg["train"]
    data_cfg = cfg_train["data"]

    valid_jsonl = data_cfg["valid_data_path"]
    max_tokens = args.max_tokens or int(data_cfg.get("max_tokens", 1024))
    substrate_seq_map, substrate_pdb_map = load_substrate_map(args.substrate_db)

    # 2) 加载模型 + 权重
    model = load_model(cfg_model, args.ckpt, device)

    # 3) pocket 阈值：用 config 里的 pba.pocket_threshold
    pocket_thr = float(cfg_model["pba"].get("pocket_threshold", 0.03))

    # 4) 输出目录：默认保存在 ckpt 同目录
    if args.out:
        out_dir = args.out
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        out_dir = os.path.join(ckpt_dir, f"val_generation_{ckpt_name}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Write per-sample predictions to dir: {out_dir}")

    global_index = 0

    with torch.no_grad():
        # 5) 遍历 val 数据
        batch_iter = (
            jsonl_single_iterator(valid_jsonl)
            if args.gen_batch_size == 1
            else jsonl_batch_iterator(valid_jsonl, max_tokens=max_tokens)
        )
        for batch in batch_iter:
            # 预处理 ec4，避免格式异常导致模型里 int 转换报错
            batch = [
                {
                    **sample,
                    "ec4": normalize_ec4(sample.get("ec4", "0.0.0.0")),
                }
                for sample in batch
            ]

            # print(f"[INFO] Processing batch starting at global index {global_index}, batch size {len(batch)}")
            # --- 5.1 使用 PEGM.init_data 做预处理 ---

            # print(f"[INFO] Processing batch starting at global index {global_index}, batch size {len(batch)}")

            batch_data = model.init_data(batch)

            # 拿到 forward 需要的字段
            src_tokens = batch_data["src_tokens"].to(device)        # [B,L]
            src_lengths = batch_data["src_lengths"].to(device)      # [B]
            res_padding = batch_data["res_padding"].to(device)      # [B,L]
            coords_in = batch_data["coords"].to(device)             # [B,L,3] (masked 版本)
            mask = batch_data["mask"].to(device)                    # [B,L]
            ec1 = batch_data["ec1"].to(device)
            ec2 = batch_data["ec2"].to(device)
            ec3 = batch_data["ec3"].to(device)
            ec4 = batch_data["ec4"].to(device)
            lig_coords_in = batch_data["lig_coords"].to(device)     # [B,L1,3]
            lig_feats_in = batch_data["lig_feats"].to(device)       # [B,L1,5]
            lig_padding = batch_data["lig_padding"].to(device)      # [B,L1]

            # 保留一些 GT 信息（在 CPU 上就行）
            coords_wo_mask = batch_data["coords_wo_mask"]           # [B,L,3]
            src_tokens_wo_mask = batch_data["src_tokens_wo_mask"]   # [B,L]
            pocket_idxs_gt_batch = batch_data["pocket_idxs"]        # List[List[int]]

            # --- 5.2 前向推理 ---
            pegm_out = model(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                res_padding=res_padding,
                coords=coords_in,
                mask=mask,
                ec1=ec1, ec2=ec2, ec3=ec3, ec4=ec4,
                lig_coords=lig_coords_in,
                lig_feats=lig_feats_in,
                lig_padding=lig_padding,
            )

            enzyme_pred = pegm_out["enzyme_pred"]   # dict: {"logits": ..., "prob": ...}
            logits = enzyme_pred["logits"]          # [B, L, 20]
            res_coords_pred = pegm_out["res_coords"]  # [B, L, 3]
            s_i_g = pegm_out["s_i_g"]              # [B, L], 口袋 score
            res_padding_b = res_padding            # [B, L]

            # print("s_i_g shape : ", s_i_g.shape)

            # --- 5.3 序列预测：logits → 氨基酸序列 ---
            #   使用你在 PEGM 里写好的 pred_to_seq
            pred_seqs = model.pred_to_seq(
                logits, src_lengths, src_tokens_wo_mask.to(device), mask
            )  # List[str]，长度 B

            B = src_tokens.size(0)

            # --- 5.4 遍历 batch 中每个样本，组装 JSON 记录 ---
            for b in range(B):
                L_real = int(src_lengths[b].item())

                # ===== 5.4.1 Ground truth 部分 =====
                sample = batch[b]
                mask_b = mask[b, :L_real]                   # [L_real]

                # 原始蛋白序列（和训练输入一致）
                if isinstance(sample["seqs"], (list, tuple)):
                    protein_seq_gt = sample["seqs"][0]
                else:
                    protein_seq_gt = sample["seqs"]
                substrate_gt = substrate_seq_map.get(protein_seq_gt)
                if substrate_gt is None:
                    pdbs_val = sample.get("pdbs") or sample.get("pdb")
                    if isinstance(pdbs_val, (list, tuple)):
                        if len(pdbs_val) > 0:
                            pdb_key = normalize_pdb_id(pdbs_val[0])
                            substrate_gt = substrate_pdb_map.get(pdb_key)
                    elif pdbs_val is not None:
                        pdb_key = normalize_pdb_id(pdbs_val)
                        substrate_gt = substrate_pdb_map.get(pdb_key)

                # 残基真实坐标：用 coords_wo_mask，截断到真实长度
                res_coords_gt = tensor_to_list(coords_wo_mask[b, :L_real, :])

                # GT 口袋编号（数据里直接给的）
                pocket_idxs_gt = sample.get("pocket_idxs", [])

                # motifs（位置列表）
                motifs_gt = sample.get("motifs", [])

                # 配体坐标 / 特征，按 lig_padding 还原真实长度
                lig_pad_b = lig_padding[b]        # [L1_max]
                lig_valid = (~lig_pad_b).bool()
                lig_coords_gt = tensor_to_list(lig_coords_in[b][lig_valid])
                lig_feats_gt = tensor_to_list(lig_feats_in[b][lig_valid])

                # 配体序列：根据你数据里字段名自行调整
                ligand_seq_gt = None
                if "ligand_seq" in sample:
                    ligand_seq_gt = sample["ligand_seq"]
                elif "ligand_seqs" in sample:
                    lig_tmp = sample["ligand_seqs"]
                    if isinstance(lig_tmp, (list, tuple)) and len(lig_tmp) > 0:
                        first = lig_tmp[0]
                        # 若仍是单元素列表，继续展开
                        if isinstance(first, (list, tuple)) and len(first) == 1 and isinstance(first[0], str):
                            ligand_seq_gt = first[0]
                        elif isinstance(first, str):
                            ligand_seq_gt = first

                # ===== 5.4.2 Prediction 部分 =====
                # 预测蛋白序列
                protein_seq_pred = pred_seqs[b]

                # 预测残基坐标
                # 未被 mask 的位置直接用 GT 坐标，只有 mask 位置用预测，避免未掩码位被改写
                res_coords_pred_b = torch.where(
                    mask_b.unsqueeze(-1),
                    res_coords_pred[b, :L_real, :],
                    coords_wo_mask[b, :L_real, :].to(res_coords_pred.device),
                )
                res_coords_pred_b = tensor_to_list(res_coords_pred_b)

                # 预测口袋编号：基于 s_i_g + 阈值
                if s_i_g is not None:
                    s_b = s_i_g[b].squeeze(-1)         # [L]
                    # 忽略 padding 位置，只在非 padding 上做阈值
                    valid_mask = ~res_padding_b[b]     # [L]
                    pocket_mask_pred = (s_b > pocket_thr) & valid_mask
                    pocket_idxs_pred = torch.nonzero(
                        pocket_mask_pred[:L_real], as_tuple=False
                    ).reshape(-1).detach().cpu().tolist()
                else:
                    # 如果 ablation 关闭了 PBA，没有 s_i_g，就不给预测口袋
                    pocket_idxs_pred = []

                rec = {
                    "index": global_index,
                    # 保留原始 ec4（你之后 eval 如果用得上）
                    "ec4": sample.get("ec4", None),

                    "ground_truth": {
                        "protein_seq": protein_seq_gt,
                        "res_coords": res_coords_gt,
                        "pocket_idxs": pocket_idxs_gt,
                        "motifs": motifs_gt,
                        "ligand_coords": lig_coords_gt,
                        "ligand_seq": ligand_seq_gt,
                        "ligand_feats": lig_feats_gt,
                        "substrate": substrate_gt,
                    },

                    "prediction": {
                        "protein_seq": protein_seq_pred,
                        "res_coords": res_coords_pred_b,
                        "pocket_idxs": pocket_idxs_pred,
                    },
                }

                out_file = os.path.join(out_dir, f"{global_index:06d}.json")
                with open(out_file, "w", encoding="utf-8") as fw:
                    json.dump(rec, fw)
                global_index += 1

    print(f"[INFO] Done. Total samples: {global_index}")
    print(f"[INFO] Output saved to dir: {out_dir}")


if __name__ == "__main__":
    main()
