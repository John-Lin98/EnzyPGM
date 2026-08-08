import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NAELayer import NAELayer
from models.SNELayer import SpatialNeighborhoodEquivariantLayer
from models.PBALayer import PocketEnhancedBilevelAttentionLayer
from models.EnzymePrediction import EnzymePrediction

def _pad_to_batch_max(x: torch.Tensor, lengths: torch.Tensor, pad_value: float = 0.0, dim: int = 1):
    """将 x 在维度 dim 上右侧补到本 batch 最大真实长度，返回 (x_pad, pad_mask[True=pad])"""
    assert x.size(0) == lengths.size(0)
    L_max = int(lengths.max().item())
    cur_L = x.size(dim)

    if cur_L < L_max:
        pad_sizes = [0] * (2 * x.dim())
        idx = (x.dim() - dim - 1) * 2 + 1
        pad_sizes[idx] = L_max - cur_L
        x = F.pad(x, pad_sizes, value=pad_value)
    elif cur_L > L_max:
        sl = [slice(None)] * x.dim()
        sl[dim] = slice(0, L_max)
        x = x[tuple(sl)]

    ar = torch.arange(L_max, device=x.device).unsqueeze(0)
    pad_mask = ar >= lengths.view(-1, 1)
    return x, pad_mask


import torch
import torch.nn as nn
from typing import Tuple


class ResidueMaskedLayerNorm(nn.Module):
    """
    Args:
        hidden_dim:  E
        eps:
        affine:      是否学习仿射参数 gamma/beta
        zero_pad:    是否将 padding 位置显式置零

    Inputs:
        res_feat_out: [B, L, E]  PBALayer 的残基特征输出
        res_padding : [B, L]     True=padding 位置

    Returns:
        res_feat_norm: [B, L, E]  仅非 padding 位置被归一化
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5, affine: bool = True, zero_pad: bool = False, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.ln = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=affine).to(self.device)
        self.zero_pad = zero_pad

    def forward(self, res_feat_out: torch.Tensor, res_padding: torch.Tensor) -> torch.Tensor:
        assert res_feat_out.dim() == 3, "res_feat_out should be [B, L, E]"
        assert res_padding.dim() == 2, "res_padding should be [B, L]"
        B, L, E = res_feat_out.shape

        # 布尔 mask：True = 非 padding（需要归一化）
        valid = (~res_padding.bool()).view(B, L)

        # 复制一份作为输出，避免对输入做 in-place
        out = res_feat_out.clone()

        if valid.any():
            # 选出非 padding 的条目，形状展平为 [N_valid, E]，做 LayerNorm
            x_valid = res_feat_out[valid]                  # [N_valid, E]
            out_valid = self.ln(x_valid)                   # [N_valid, E]
            out[valid] = out_valid                         # 写回对应位置

        # 可选：把 padding 位置清零
        if self.zero_pad:
            out = out.masked_fill(res_padding.unsqueeze(-1), 0.0)

        return out


class PocketAugmentedEnzymeGenerativeModel(nn.Module):
    """
    Pocket-augmented Enzyme Generative Model (PEGM)
    ------------------------------------------------
    - self.nae : NAELayer（内部集成 RFF + 残基侧 SNE/EGNN）
    - self.sne : SpatialNeighborhoodEquivariantLayer（配体侧）
    - self.pba : PocketEnhancedBilevelAttentionLayer（残基-配体双层注意力）
    - self.enz_pred : EnzymePrediction（逐位残基类型）
    - self.bind_head : Pocket-Substrate Binding Strength 预测头（预留）
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.get("pegm", {})
        self.pad_idx = int(mcfg.get("pad_idx", 1))
        self.embeding_dim = int(mcfg.get("embeding_dim"))
        self.pba_layer_num = int(mcfg.get('pba_layer_num', 3))
        self.device = torch.device(mcfg.get('device', 'cpu'))
        self.ablation = cfg.get('ablation', None)

        # --- 子模块 ---
        self.nae = NAELayer(cfg.get("nae"))
        self.sne = SpatialNeighborhoodEquivariantLayer(cfg.get("sne"))
        self.pba_layer = nn.ModuleList()
        self.pba_norm = nn.ModuleList()

        for i in range(self.pba_layer_num):
            self.pba_layer.append(PocketEnhancedBilevelAttentionLayer(cfg.get("pba")))
            self.pba_norm.append(ResidueMaskedLayerNorm(hidden_dim=self.embeding_dim, device=self.device))

        self.enz_pred = EnzymePrediction(cfg.get("enzyme_head"))

        self.bind_head = None

    # ========= 这里是你要的预处理入口 =========
    def init_data(self, batch: list) -> dict:
        """
        batch: List[dict]，每个 dict 至少包含：
            'seqs', 'coords', 'ec4', 'ligand_coords', 'ligand_feats', 'motifs'
        返回：
            {
              src_tokens: [B,L], src_lengths: [B], res_padding: [B,L], mask: [B,L],
              coords: [B,L,3],
              ec1,ec2,ec3,ec4: [B],
              ligand_coords: [B,L1,3], ligand_feats: [B,L1,5], ligand_padding: [B,L1]
            }
        """
        device = next(self.parameters()).device

        # ===== 0) 取 alphabet & MASK id =====
        alphabet = None
        mask_idx = None
        try:
            # 常见挂载路径（按你给的 NAELayer -> RFFL -> ESM）
            alphabet = self.nae.rffl.alphabet
            mask_idx = getattr(alphabet, "mask_idx", None)
            if mask_idx is None and hasattr(alphabet, "tokens_to_ids"):
                mask_idx = alphabet.tokens_to_ids.get("<mask>", None)
        except Exception:
            pass
        if alphabet is None or mask_idx is None:
            raise RuntimeError("ESM alphabet 未就绪：请确保 self.nae.rffl.alphabet 存在且含 mask_idx")

        # batch_converter = alphabet.get_batch_converter()

        B = len(batch)

        # ===== 1) 序列 → token ids；补齐到 batch 内最长；生成 padding_mask & src_lengths =====
        # 取 alphabet 与必要 id
        alphabet = self.nae.rffl.alphabet
        pad_idx = getattr(alphabet, "padding_idx", self.pad_idx)
        # 兼容不同版本字段名
        if hasattr(alphabet, "tok_to_idx"):
            tok2id = alphabet.tok_to_idx
        elif hasattr(alphabet, "tokens_to_ids"):
            tok2id = alphabet.tokens_to_ids
        else:
            raise RuntimeError("alphabet 未提供 tok_to_idx / tokens_to_ids")

        unk_idx = tok2id.get("<unk>", 0)

        # 原始大写序列（不插入 BOS/EOS）
        seq_list = [ (sample["seqs"][0] if isinstance(sample["seqs"], (list, tuple)) else sample["seqs"]).upper()
                    for sample in batch ]

        # 每条序列逐字符映射
        id_tensors = []
        for s in seq_list:
            ids = [tok2id.get(ch, unk_idx) for ch in s]
            id_tensors.append(torch.tensor(ids, dtype=torch.long))

        # 真实长度（就是原序列长度）
        src_lengths = torch.tensor([t.numel() for t in id_tensors], dtype=torch.long, device=device)
        L_max = int(src_lengths.max().item())

        # 右侧补 PAD 到 batch 最大长度
        B = len(id_tensors)
        src_tokens = torch.full((B, L_max), fill_value=pad_idx, dtype=torch.long, device=device)
        for i, t in enumerate(id_tensors):
            n = t.numel()
            if n > 0:
                src_tokens[i, :n] = t.to(device)

        # padding mask（True=PAD）
        ar = torch.arange(L_max, device=device).unsqueeze(0)  # [1, L_max]
        res_padding = ar >= src_lengths.view(-1, 1)

        # ESM 的 batch_converter 需要形如 [(label, seq), ...] 的列表；label 可随便给
        # batch_tuples = [(str(i), (sample["seqs"][0]).upper()) for i, sample in enumerate(batch)]
        # labels, strs, tokens = batch_converter(batch_tuples)   # tokens: [B, Lmax]
        # tokens = tokens.to(next(self.parameters()).device)

        # # padding mask（True=pad）直接由 token 等于 pad 得到
        # pad_idx = getattr(alphabet, "padding_idx", self.pad_idx)
        # res_padding = tokens.eq(pad_idx)

        # # 真实长度（不含 PAD；注意 batch_converter 默认会在首尾加特殊符号）
        # bos_idx = getattr(alphabet, "bos_idx", getattr(alphabet, "cls_idx", None))
        # eos_idx = getattr(alphabet, "eos_idx", None)

        # # 如果存在 BOS/EOS，则真实氨基酸长度 = 非 PAD 的位置数 - (BOS?1:0) - (EOS?1:0)
        # nonpad_len = (~res_padding).sum(dim=1)  # 包含特殊符号
        # specials = (1 if bos_idx is not None else 0) + (1 if eos_idx is not None else 0)
        # src_lengths = (nonpad_len - specials).to(torch.long).clamp_min(0)

        # src_tokens = tokens  # 之后把它传给 NAELayer；它已经是按 batch 最大长度补好的

        # print(res_padding[0][0], res_padding[0][-1])

        _, L_max = src_tokens.shape

        # print('!!!!!!!!!')
        # print(batch)
        # ===== 1) 序列 → token ids；补齐到 batch 内最长；生成 padding_mask & src_lengths =====
        # seqs = [sample["seqs"] for sample in batch]  # List[str]

        # # print(seqs)

        # B = len(seqs)
        # # 将每条序列转成 ids（不依赖 batch_converter，直接用 tok_to_idx）
        # if hasattr(alphabet, "tok_to_idx"):
        #     tok2id = alphabet.tok_to_idx
        # elif hasattr(alphabet, "tokens_to_ids"):
        #     tok2id = alphabet.tokens_to_ids
        # else:
        #     raise RuntimeError("alphabet 未提供 tok_to_idx / tokens_to_ids")

        # def _encode_one(seq: str):
        #     # ESM2 常见：需要在首尾加 <cls>/<eos>？这里与训练一致即可；若 NAELayer 内部自己加，就不要重复加。
        #     # 这里假设直接逐字符映射（如 'A','C','D',...'X'），未知映射到 '<unk>'
        #     ids = []
        #     for ch in seq:
        #         # print('ch', ch)
        #         ids.append(tok2id.get(ch, tok2id.get("<unk>", 0)))
        #     return torch.tensor(ids, dtype=torch.long)

        # ids_list = [ _encode_one(s) for s in seqs ]  # List[T(L_i)]



        # # print(ids_list)

        # src_lengths = torch.tensor([t.numel() for t in ids_list], dtype=torch.long, device=device)
        # L_max = int(src_lengths.max().item())
        # src_tokens = torch.full((B, L_max), fill_value=self.pad_idx, dtype=torch.long, device=device)
        # for i, t in enumerate(ids_list):
        #     n = t.numel()
        #     src_tokens[i, :n] = t.to(device)
        # # padding mask（True=pad）
        # ar = torch.arange(L_max, device=device).unsqueeze(0)
        # res_padding = ar >= src_lengths.view(-1, 1)

        # print('src_tokens', src_tokens)
        # print('res_padding', res_padding)
        # print('src_lengths', src_lengths)


        # ===== 2) ec4 "a.b.c.d" → 4×int =====
        ec_list = [sample["ec4"] for sample in batch]  # List[str]
        ec_parts = []
        ec1, ec2, ec3, ec4 = [], [], [], []
        for s in ec_list:
            # 容错：可能是 "2.5.1.18" 或 "EC 2.5.1.18"
            s = str(s).strip().split()[-1]
            # print('s : ', s)
            xs = s.split(".")
            # print('xs. : ', xs)
            # four = []

            ec1.append(int(xs[0]))
            ec2.append(int(xs[1]))
            ec3.append(int(xs[2]))
            ec4.append(int(xs[3]))
        # ec_parts = torch.tensor(ec_parts, dtype=torch.long, device=device)  # [B,4]
        # ec1, ec2, ec3, ec4 = [ec_parts[:, i].contiguous() for i in range(4)]
        ec1 = torch.tensor(ec1, dtype=torch.long, device=device)
        ec2 = torch.tensor(ec2, dtype=torch.long, device=device)
        ec3 = torch.tensor(ec3, dtype=torch.long, device=device)
        ec4 = torch.tensor(ec4, dtype=torch.long, device=device)
        # print('ec', ec1)

        # ===== 3) coords 整理为 [B,L,3]，补齐到与 src_tokens 相同长度 =====
        #  coords 是三层嵌套，取第 0 维就是每条序列的坐标序列
        # for sample in batch:
            # print('sample["coords"]', sample["coords"])
        coords_list = [ torch.tensor(sample["coords"], dtype=torch.float32) for sample in batch ]  # each [Li,3] or [Li,*,3]
        # print(coords_list[0].shape)
        # print(coords_list[1].shape)
        # 压到 [Li,3]
        # print('coords_list : ', coords_list)
        coords_list = [ c.view(-1, 3) for c in coords_list ]

        coords = torch.zeros((B, L_max, 3), dtype=torch.float32, device=device)
        # print('coords : ', coords)

        for i, c in enumerate(coords_list):
            n = min(c.size(0), L_max)
            coords[i, :n] = c[:n].to(device)

        # print(coords.shape)
        # print(coords[0])
        # print(coords)
        # print('motifs : ', batch[0].get("motifs"))
        # ===== 4) motifs → mask [B,L]；根据 mask 覆盖 seq/coords =====
        # motifs 为位置索引列表（从 0 开始假设）。超长索引会被忽略。
        mask = torch.zeros((B, L_max), dtype=torch.bool, device=device)
        for i, sample in enumerate(batch):
            idxs = sample.get("motifs", []) or []
            for j in idxs:
                if 0 <= j < L_max:
                    mask[i, j] = True

        # print('readed mask : ', mask)
        # print(mask_idx)

        # 覆盖：coords 对应位置设为 0；tokens 设为 MASK
        # for i in range(B):
        #     for j in range(L_max):
        #         if mask[i, j] == True:
        #             coords[i, j] = torch.tensor([0, 0, 0], dtype=coords.dtype, device=coords.device)
        #             src_tokens[i, j] = torch.float32(0, dtype=src_tokens.dtype, device=src_tokens.device)

        # coords = torch.where(mask.unsqueeze(-1), torch.zeros_like(coords), coords)
        src_tokens_wo_mask = src_tokens.clone()
        coords_wo_mask = coords.clone()

        # print('mask' , mask)
        coords = torch.where(mask.unsqueeze(-1), torch.zeros_like(coords), coords)
        src_tokens = torch.where(mask, torch.full_like(src_tokens, mask_idx), src_tokens)
        # src_tokens = torch.where(mask, torch.full_like(src_tokens, mask_idx), src_tokens)

        # print(coords)
        # print(src_tokens)

        # ===== 5) ligand_coords → [B,L1,3]（按 batch 内最长补齐），ligand_padding =====
        lig_coords_list = [ torch.tensor(s["ligand_coords"], dtype=torch.float32) for s in batch ]  # each [Ml,3]
        # print(lig_coords_list[0].shape)
        # print(lig_coords_list[1].shape)
        lig_lengths = torch.tensor([c.view(-1, 3).size(0) for c in lig_coords_list], dtype=torch.long, device=device)
        L1_max = int(lig_lengths.max().item()) if len(lig_coords_list) > 0 else 0
        if L1_max > 0:
            lig_coords = torch.zeros((B, L1_max, 3), dtype=torch.float32, device=device)
            for i, c in enumerate(lig_coords_list):
                c = c.view(-1, 3)
                n = min(c.size(0), L1_max)
                lig_coords[i, :n] = c[:n].to(device)
            ar1 = torch.arange(L1_max, device=device).unsqueeze(0)
            ligand_padding = ar1 >= lig_lengths.view(-1, 1)
        else:
            lig_coords = torch.zeros((B, 0, 3), dtype=torch.float32, device=device)
            ligand_padding = torch.zeros((B, 0), dtype=torch.bool, device=device)

        # print('lig_coords', lig_coords)
        # print('ligand_padding', ligand_padding)

        # ===== 6) ligand_feats → [B,L1,5] 同步补齐 =====
        lig_feats_list = [ torch.tensor(s["ligand_feats"], dtype=torch.float32) for s in batch ]  # each [Ml,5]
        # print("!!!!!!!!!")
        # print(lig_feats_list[0].shape)
        # print(lig_feats_list[1].shape)
        # print("!!!!!!!!!")

        if L1_max > 0:
            lig_feats = torch.zeros((B, L1_max, 5), dtype=torch.float32, device=device)
            for i, f in enumerate(lig_feats_list):
                f = f.view(-1, 5)
                n = min(f.size(0), L1_max)
                lig_feats[i, :n] = f[:n].to(device)
        else:
            lig_feats = torch.zeros((B, 0, 5), dtype=torch.float32, device=device)

        # print(lig_feats)
        # print(lig_feats.shape)
        # print('lig_padding : ',ligand_padding)
        # print('res_padding : ', res_padding)


        return {
            "src_tokens": src_tokens,            # [B, L]
            "src_lengths": src_lengths,          # [B]
            "res_padding": res_padding,          # [B, L]
            "coords": coords,                    # [B, L, 3]
            "mask": mask,                        # [B, L] 重要位点 mask
            "ec1": ec1, "ec2": ec2, "ec3": ec3, "ec4": ec4,  # [B]
            "lig_coords": lig_coords,         # [B, L1, 3]
            "lig_feats": lig_feats,           # [B, L1, 5]
            "lig_padding": ligand_padding,     # [B, L1]
            "pocket_idxs": [sample['pocket_idxs'] for sample in batch], # [[int, int, ...], [int, int, ...]]
            "src_tokens_wo_mask": src_tokens_wo_mask,
            "coords_wo_mask": coords_wo_mask
        }
    def forward(self,
        src_tokens,
        src_lengths,
        res_padding,
        coords,
        mask,
        ec1,
        ec2,
        ec3,
        ec4,
        lig_coords,
        lig_feats,
        lig_padding
    ):



        nae_fusion_out = self.nae(src_tokens, src_lengths, coords, mask, ec1, ec2, ec3, ec4, res_padding)

        res_feats = nae_fusion_out['hidden']
        res_coords = nae_fusion_out['coords']

        if check_nan(res_coords):
            print('NaN in res_coords after NAE')
            exit(-1 )

        fusion_feat, lig_feats, lig_coords, lig_padding = self.sne(lig_feats, lig_coords, lig_padding) # lig_feats [B, L, 5] -> [B, L, H]

        if check_nan(lig_coords):
            print('NaN in lig_coords after SNE')
            exit(-1 )



        if self.ablation.get('wo_PBA', False) == True:

            print("Pass PBA")
            # 直接跳过 PBA 层
            enzyme_pred = self.enz_pred(res_feats, torch.zeros_like(res_padding).bool(), res_padding)    # {"logits": "", "prob": "(after softmax)"}

            if check_nan(enzyme_pred['logits']):
                print('NaN in enzyme_pred logits after EnzymePrediction (wo_PBA)')
                exit(-1 )

            pegm_fusion_out = {
                "enzyme_pred":enzyme_pred,
                "res_feats": res_feats,
                "res_coords": res_coords,
                "pocket_idx": None,
                "pocket_mask": None,
                "lig_feats": lig_feats,
                "lig_coords": lig_coords,
                "res_padding": res_padding,
                "lig_padding": lig_padding,
                "mask": mask,
                "s_i_g": None,
            }
            return pegm_fusion_out



        for i in range(self.pba_layer_num):
            pba_layer, layernorm = self.pba_layer[i], self.pba_norm[i]

            res_feats, res_coords, pocket_idx, pocket_mask, lig_feats, lig_coords, logits_rl, s_i_g = \
                pba_layer(res_feats, res_coords, lig_feats, lig_coords, res_padding, lig_padding)

            res_feats = layernorm(res_feats, res_padding)

        if check_nan(res_coords):
            print('NaN in res_coords after PBA')
            exit(-1 )

        enzyme_pred = self.enz_pred(res_feats, pocket_mask, res_padding)    # {"logits": "", "prob": "(after softmax)"}

        if check_nan(enzyme_pred['logits']):
            print('NaN in enzyme_pred logits after EnzymePrediction')
            exit(-1 )

        if self.bind_head is not None:                                      # binding prediction
            pass



        pegm_fusion_out = {
            "enzyme_pred":enzyme_pred,
            "res_feats": res_feats,
            "res_coords": res_coords,
            "pocket_idx": pocket_idx,
            "pocket_mask": pocket_mask,
            "lig_feats": lig_feats,
            "lig_coords": lig_coords,
            "res_padding": res_padding,
            "lig_padding": lig_padding,
            "mask": mask,
            "s_i_g": s_i_g,
        }
        return pegm_fusion_out


    # def pred_to_seq(self, pred):
    def pred_to_seq(self, pred, src_lengths, src_tokens=None, mask=None):
        """
        pred: LongTensor [B, L, V]
        src_lengths: [B]
        src_tokens: Optional LongTensor [B, L] 原始未mask的token ids，用于未mask位置复原
        mask: Optional BoolTensor [B, L]，True 表示该位置被mask
        返回: List[str]，长度为 B
        """
        alphabet = getattr(self.nae.rffl, "alphabet", None)
        if alphabet is None or not hasattr(alphabet, "get_tok"):
            raise RuntimeError("alphabet 不可用，无法解码预测序列")

        get_tok = alphabet.get_tok
        ids20 = pred.argmax(dim=-1)             # [B, L]
        B, L = ids20.shape
        seqs = []
        has_src = src_tokens is not None and mask is not None

        for i in range(B):
            L_real = int(src_lengths[i])
            toks = []
            for j in range(L_real):
                use_pred = True
                if has_src:
                    # mask True 表示被遮盖，需要用预测；否则用原 token
                    use_pred = bool(mask[i, j])
                if use_pred:
                    tok = get_tok(int(ids20[i, j]) + 4)  # +4 对应 alphbet 中的第一个氨基酸
                else:
                    tok = get_tok(int(src_tokens[i, j]))
                if isinstance(tok, bytes):
                    tok = tok.decode()
                if isinstance(tok, str) and tok.startswith("<"):
                    continue
                toks.append(tok)
            seqs.append("".join(toks))
        return seqs


def check_nan(x):
    if torch.isnan(x).any():
        return True
    else:
        return False
