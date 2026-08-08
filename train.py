# train.py
import os, json, time, random
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import EnzymeLigandDataset
from models.PEGM import PocketAugmentedEnzymeGenerativeModel
from models.criterions.PocketEnhancedLoss import PocketEnhancedLoss
from utils.ckpt import save_checkpoint, load_for_resume
from torch.utils.tensorboard import SummaryWriter

from utils.Logger import logging, setup_logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 0.01)
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = cfg.get("eps", 1e-8)
    return optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    sched = cfg.get("scheduler", "none").lower()
    if sched == "cosine":
        T = cfg.get("epochs", 10)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    if sched == "step":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    return None


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)


    log_path = cfg['train']['log_path']
    log_parent = Path(log_path).parent
    log_parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).touch(exist_ok=True)

    setup_logging(log_file=log_path)
    # ========= 训练配置缺省 =========
    train_cfg = cfg.get("train", {})
    data_cfg  = train_cfg.get("data", {})
    crit_cfg  = train_cfg.get("criterion", {})
    model_cfg = cfg.get("model", {})
    log_dir   = Path(train_cfg.get("log_path", "./logs"))
    if not os.path.exists(log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)

    epochs         = train_cfg.get("epochs", 10)
    device_str     = cfg.get("model", {}).get("pegm", {}).get("device", "cuda")
    device         = torch.device(device_str if torch.cuda.is_available() else "cpu")
    seed           = data_cfg.get("seed", 1234)
    amp_enabled    = train_cfg.get("amp", False)
    grad_clip      = train_cfg.get("grad_clip", 1.0)
    save_every_ep  = train_cfg.get("save_every", 1)
    print_every    = train_cfg.get("print_every", 50)
    resume_dir     = train_cfg.get("resume_dir", str(log_dir / "checkpoints"))  # 目录或具体文件
    ckpt_dir       = Path(train_cfg.get("ckpt_dir", str(log_dir / "checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)



    # ========= 数据 =========
    dataset_train = EnzymeLigandDataset(
        cfg=data_cfg
    )
    dataset_val = EnzymeLigandDataset(
        cfg=data_cfg, is_valid_set=True
    )

    train_num_batches = sum(1 for _ in dataset_train.get_epoch_iterator())
    val_num_batches = sum(1 for _ in dataset_val.get_epoch_iterator())

    print(f'train len : {len(dataset_train)}')
    print(f'val len : {len(dataset_val)}')
    # exit()

    # ========= 模型 & 损失 =========
    model = PocketAugmentedEnzymeGenerativeModel(cfg["model"]).to(device)
    criterion = PocketEnhancedLoss(crit_cfg).to(device)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # ========= 断点续训（自动从目录取最新）=========
    resume_path = Path(resume_dir)
    if resume_path.exists():
        start_epoch, start_step, used_ckpt = load_for_resume(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_dir_or_file=resume_dir,   # 可传目录（自动找最新）或具体文件
            map_location=device,
            strict=True,
        )
    else:
        start_epoch, start_step, used_ckpt = None, None, None
    if used_ckpt is not None:
        print(f"[resume] loaded from: {used_ckpt} (epoch={start_epoch}, step={start_step})")
    else:
        start_epoch = 0
        start_step  = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=train_cfg.get('tensorboard_dir'), purge_step=start_epoch)
    # ========= 训练循环 =========
    global_step = start_step
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        t0 = time.time()

        iterator = dataset_train.get_epoch_iterator()

        running = {"loss": 0.0}

        loss_sum = {
            "loss": 0.0,
            "loss_lm": 0.0,
            "loss_coord": 0.0,
            "loss_pocket_gate": 0.0,
            "loss_pocket_coord": 0.0
        }

        for it, batch in enumerate(iterator, start=1):

            # print(f'batch : {batch}')

            # for b in batch:
            #     print('ec4 : ', b['ec4'])

            batch = model.init_data(batch)

            # for b in batch:
            #     print('after init_data ec4 : ', b['ec4'])


            batch = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # 你的模型：接收 batch（已是 init_data 格式），返回 pegm_fusion_out

                # print(batch)

                out = model(
                    src_tokens=batch['src_tokens'],
                    src_lengths=batch['src_lengths'],
                    res_padding=batch['res_padding'],
                    coords=batch['coords'],
                    mask=batch['mask'],
                    ec1=batch['ec1'],
                    ec2=batch['ec2'],
                    ec3=batch['ec3'],
                    ec4=batch['ec4'],
                    lig_coords=batch['lig_coords'],
                    lig_feats=batch['lig_feats'],
                    lig_padding=batch['lig_padding']
                )
                loss_dict = criterion(out, batch)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()

            # print([name for name, _ in model.named_modules() if "pba" in name.lower()])
            # print([name for name, _ in model.named_parameters() if "pba" in name.lower()])
            # coord_params = []
            # for name, p in model.named_parameters():
            #     # print(f' name : {str(name)}')
            #     if "coord" in name.lower() or "egnn" in name.lower() or "pba" in name.lower():
            #         coord_params.append((name, p))

            # for name, p in coord_params:  # 打印前 10 个看看
            #     if p.grad is None:
            #         print("[NO GRAD]", name)
            #     else:
            #         print("[GRAD]", name, p.grad.norm().item())

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 每 step 调整一次
                scheduler.step()

            running["loss"] += float(loss.detach().item())
            global_step += 1

            writer.add_scalar('train/loss', loss_dict["loss"].detach().item(), global_step)
            writer.add_scalar('train/loss_lm', loss_dict["loss_lm"].detach().item(), global_step)
            writer.add_scalar('train/loss_coord', loss_dict["loss_coord"].detach().item(), global_step)
            writer.add_scalar('train/loss_pocket_gate', loss_dict["loss_pocket_gate"].detach().item(), global_step)
            writer.add_scalar('train/loss_pocket_coord', loss_dict["loss_pocket_coord"].detach().item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], global_step)


            if it % print_every == 0:
                avg = running["loss"] / print_every
                lr  = optimizer.param_groups[0]["lr"]
                msg = f"[epoch {epoch}/{epochs}] | step {it:6d}/{train_num_batches} | global_step {global_step:6d} | loss {avg:.6f} | lr {lr:.3e}"
                # 若你在 loss_dict 里有分项（lm/coord/pocket_*），也打出来
                for k in ["loss_lm", "loss_coord", "loss_pocket_gate", "loss_pocket_coord"]:
                    if k in loss_dict:
                        msg += f" | {k} {float(loss_dict[k]):.4f}"
                print(msg)
                running["loss"] = 0.0

            loss_sum["loss"] += float(loss_dict["loss"].detach().item())
            loss_sum["loss_lm"] += float(loss_dict["loss_lm"].detach().item())
            loss_sum["loss_coord"] += float(loss_dict["loss_coord"].detach().item())
            loss_sum["loss_pocket_gate"] += float(loss_dict["loss_pocket_gate"].detach().item())
            loss_sum["loss_pocket_coord"] += float(loss_dict["loss_pocket_coord"].detach().item())


            if it % 100 == 0:
                avg_loss = loss_sum["loss"] / 100
                avg_loss_lm = loss_sum["loss_lm"] / 100
                avg_loss_coord = loss_sum["loss_coord"] / 100
                avg_loss_pocket_gate = loss_sum["loss_pocket_gate"] / 100
                avg_loss_pocket_coord = loss_sum["loss_pocket_coord"] / 100

                writer.add_scalar('train/avg_loss_per100steps', avg_loss, global_step)
                writer.add_scalar('train/avg_loss_lm_per100steps', avg_loss_lm, global_step)
                writer.add_scalar('train/avg_loss_coord_per100steps', avg_loss_coord, global_step)
                writer.add_scalar('train/avg_loss_pocket_gate_per100steps', avg_loss_pocket_gate, global_step)
                writer.add_scalar('train/avg_loss_pocket_coord_per100steps', avg_loss_pocket_coord, global_step)

                loss_sum["loss"] = 0.0
                loss_sum["loss_lm"] = 0.0
                loss_sum["loss_coord"] = 0.0
                loss_sum["loss_pocket_gate"] = 0.0
                loss_sum["loss_pocket_coord"] = 0.0


        # ======= 验证（可选）=======
        if dataset_val is not None and len(dataset_val) > 0:
            model.eval()
            with torch.no_grad():
                val_iter = dataset_train.get_epoch_iterator()

                val_loss_sum, val_steps = 0.0, 0
                vloss_lm = 0
                vloss_coord = 0
                vloss_pocket_lm = 0
                vloss_pocket_coord = 0
                vloss_dict = {}
                for vb in val_iter:
                    vb = model.init_data(vb)
                    vb = to_device(vb, device)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        vout = model(
                            src_tokens=vb['src_tokens'],
                            src_lengths=vb['src_lengths'],
                            res_padding=vb['res_padding'],
                            coords=vb['coords'],
                            mask=vb['mask'],
                            ec1=vb['ec1'],
                            ec2=vb['ec2'],
                            ec3=vb['ec3'],
                            ec4=vb['ec4'],
                            lig_coords=vb['lig_coords'],
                            lig_feats=vb['lig_feats'],
                            lig_padding=vb['lig_padding']
                        )
                        vloss_dict = criterion(vout, vb)
                        vloss = vloss_dict["loss"]
                        vloss_lm = float(vloss_dict["loss_lm"].detach().item())
                        vloss_coord = float(vloss_dict["loss_coord"].detach().item())
                        vloss_pocket_gate = float(vloss_dict["loss_pocket_gate"].detach().item())
                        vloss_pocket_coord = float(vloss_dict["loss_pocket_coord"].detach().item())

                    val_loss_sum += float(vloss.detach().item())
                    val_steps += 1
                val_avg = val_loss_sum / max(1, val_steps)
                print(( f"[validate] epoch {epoch} | val_loss {val_avg:.6f} | "
                        f"loss_lm {vloss_lm:.6f} | loss_coord {vloss_coord:.6f} | "
                        f"loss_pocket_gate {vloss_pocket_gate:.6f} | loss_pocket_coord {vloss_pocket_coord:.6f} "))
                print(vloss_dict)
                writer.add_scalar('valid/loss', vloss_dict["loss"].detach().item(), epoch)
                writer.add_scalar('valid/loss_lm', vloss_dict["loss_lm"].detach().item(), epoch)
                writer.add_scalar('valid/loss_coord', vloss_dict["loss_coord"].detach().item(), epoch)
                writer.add_scalar('valid/loss_pocket_gate', vloss_dict["loss_pocket_gate"].detach().item(), epoch)
                writer.add_scalar('valid/loss_pocket_coord', vloss_dict["loss_pocket_coord"].detach().item(), epoch)

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_avg)

        # ======= 保存 ckpt =======
        if (epoch % save_every_ep) == 0:
            # path = None
            path = save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                extra={"cfg_path": cfg_path},
                filename=f"epoch{epoch:05d}.pt",
            )
            print(f"[ckpt] saved: {path} | epoch time: {time.time()-t0:.1f}s")
    writer.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to cfg json")
    args = ap.parse_args()

    main(args.config)
