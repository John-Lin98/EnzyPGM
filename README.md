# EnzyPGM

本仓库提供论文 *EnzyPGM: Pocket-conditioned Generative Model for Substrate-specific Enzyme Design* 的原始 PEGM 实现整理版。EnzyPGM 将蛋白序列/结构、EC 编号、配体原子特征和官方口袋信息输入到 pocket-enhanced bilevel attention 模块，用于口袋条件下的酶序列与结构生成。

本次公开只包含论文实现、可编辑的配置模板、训练/推理入口和轻量评估工具。它**不包含** EnzyPGM-2、EnzyMatch、EnzyPock-2 的未发表代码或结果。

## 仓库内容

- `models/`：PEGM、残基/配体等变编码和口袋双层注意力实现。
- `train.py`：训练与 checkpoint 恢复入口。
- `generation.py`：从 checkpoint 对配置中的验证 JSONL 生成逐样本 JSON 输出。
- `scripts/evaluate_sequences.py`：对上述生成 JSON 的蛋白序列位置恢复率做轻量汇总；它不是论文表格的替代品。
- `configs/paper_main.example.json`：由论文主实验历史配置整理出的无服务器路径模板。
- `scripts/release_smoke.py`：不下载数据或权重的发布完整性检查。

## 安装

建议使用 Python 3.10 或更高版本并建立独立环境。先根据本机 CUDA/PyTorch 版本从 [PyTorch](https://pytorch.org/) 与 [PyG](https://pytorch-geometric.readthedocs.io/) 的官方说明安装 `torch` 和与之匹配的 `torch-scatter` wheel，再安装其余依赖：

```bash
pip install -r requirements.txt
```

`fair-esm` 提供 ESM-2 主干；ESM-2 权重不随本仓库分发。

## 数据与 checkpoint

EnzyPock 数据、PDB/SDF 结构、训练缓存和论文 checkpoint 均未提交到 Git。请在获得相应数据来源许可后自行准备，并在配置中填写本地相对或绝对路径。数据源的再分发和使用必须遵守其原始条款。

每个 JSONL 样本至少需要以下字段：

```json
{
  "seqs": ["PROTEIN_SEQUENCE"],
  "coords": [[[0.0, 0.0, 0.0]]],
  "ec4": ["1.1.1.1"],
  "ligand_coords": [[[0.0, 0.0, 0.0]]],
  "ligand_feats": [[[0.0, 0.0, 0.0, 0.0, 0.0]]],
  "pocket_idxs": [[0]],
  "motifs": [0]
}
```

坐标长度必须与蛋白或配体轴对应；不满足该契约的记录应在数据准备阶段拒绝，而非静默补齐。建议将可恢复的 `outputs/checkpoints/` 目录保留在 Git 之外。

## 最小检查、训练、推理与评估

先执行不依赖数据和权重的检查：

```bash
python scripts/release_smoke.py --config configs/paper_main.example.json
```

准备数据、ESM-2 权重和目标 checkpoint 后，复制配置并替换 `path/to/...` 占位符：

```bash
cp configs/paper_main.example.json my_run.json
python train.py --config my_run.json
python generation.py --config my_run.json --ckpt path/to/checkpoint.pt --out outputs/generation
python scripts/evaluate_sequences.py --predictions outputs/generation --output outputs/sequence_recovery.json
```

训练入口会在 `resume_dir` 已存在时恢复最新 checkpoint；不存在时从头训练。生成程序会使用配置中的 `valid_data_path`。完整复现需要与论文版本兼容的 EnzyPock 数据处理、ESM-2 权重和计算环境；本仓库不把这些受限/大型资产伪装成可直接下载的内容。

## 许可证与第三方组件

本仓库中 EnzyPGM 源码以 [MIT](LICENSE) 许可证发布。PyTorch、ESM-2、`torch-scatter` 与数据来源均是独立项目，必须遵守其各自许可证和访问条款；本仓库未复制它们的源码、权重或数据。详见 [第三方依赖说明](docs/THIRD_PARTY_NOTICES.md) 和 [发布来源说明](docs/RELEASE_PROVENANCE.md)。

## 引用

请引用论文 *EnzyPGM: Pocket-conditioned Generative Model for Substrate-specific Enzyme Design*。正式 BibTeX/DOI 将在论文公开元数据可核实时补充；本仓库不臆造未核实的引文信息。
