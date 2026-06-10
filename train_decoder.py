from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
import lpips
from dataset import TinyWMDataset
from model import TinyWorldModel, TinyWMDecoder

def load_config(path):
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)

def prepare_frames(frames, device):
    frames = frames.to(device, non_blocking=True)
    if frames.dtype == torch.uint8:
        return frames.float().div_(255.0) # ops on device
    return frames.float()

def make_wandb_image(pred_frames, target_frames, n_images=5):
    n_images = min(n_images, pred_frames.shape[0], target_frames.shape[0])
    pred_last = pred_frames[:n_images, -1].detach().float().cpu().clamp(0, 1)
    target_last = target_frames[:n_images, -1].detach().float().cpu().clamp(0, 1)
    comparisons = []
    for idx in range(n_images):
        comparisons.extend([target_last[idx], pred_last[idx]])
    grid_chw = make_grid(comparisons, nrow=2)
    grid_hwc = (grid_chw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return wandb.Image(grid_hwc, caption="Each row: target | prediction")

def get_dataloader(cfg, train=True):
    data_cfg = cfg["dataset"]
    loader_cfg = cfg["dataloader"]
    device = cfg["device"]
    dataset = TinyWMDataset(
        path=data_cfg["path"],
        window_len=data_cfg["window_len"],
        stride=data_cfg["stride"],
        train=train,
        n_val_episodes=data_cfg["n_val_episodes"],
        seed=cfg["seed"],
    )
    loader_kwargs = {
        "batch_size": loader_cfg["batch_size"],
        "shuffle": train,
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": (device == "cuda"),
        "drop_last": train,
    }
    if loader_cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 4)
    return DataLoader(dataset, **loader_kwargs)

def save_checkpoint(path, model, optimizer, cfg, epoch, global_step, val_loss, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )

def load_checkpoint(path, model, optimizer, device, load_optimizer=True, restore_rng=True):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if load_optimizer:
        if "optimizer" not in checkpoint:
            raise KeyError(f"Checkpoint {path} does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    # preserve rng as prev run
    if restore_rng and "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        if device == "cuda" and cuda_rng_state_all is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state_all)
            except RuntimeError as exc:
                print(f"Could not restore CUDA RNG state/ {exc}")
    return checkpoint

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/decoder_A100.yaml"))
    parser.add_argument("--wm-checkpoint", type=Path, required=True, help="Path to world model checkpoint")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume the decoder from")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader.batch_size")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--no-restore-rng", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.batch_size is not None:
        print(f"Overriding batch size; Old batch size: {cfg['dataloader']['batch_size']}; New batch size: {args.batch_size}")
        cfg["dataloader"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.resume is not None:
        cfg["checkpoint"]["resume_from"] = str(args.resume)
    resume_from = cfg["checkpoint"].get("resume_from")
    torch.manual_seed(cfg["seed"])

    out_dir = Path(cfg["checkpoint"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(cfg, train=True)
    val_loader = get_dataloader(cfg, train=False)

    # load and freeze the world model
    world_model = TinyWorldModel(cfg["TinyWorldModel"]).to(cfg["device"])
    wm_checkpoint = torch.load(args.wm_checkpoint, map_location="cpu")
    world_model.load_state_dict(wm_checkpoint["model"])
    for p in world_model.parameters():
        p.requires_grad_(False)
    world_model.eval()

    model = TinyWMDecoder(cfg["TinyWMDecoder"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=cfg["optimizer"]["lr"],
                                weight_decay=cfg["optimizer"]["weight_decay"])
    
    L1 = torch.nn.L1Loss()
    LPIPS = lpips.LPIPS(net="vgg").to(cfg["device"]).eval()
    lpips_weight = 0.1
    for p in LPIPS.parameters():
        p.requires_grad_(False)
    use_rollout = cfg["loss"]["rollout"]["use"]
    
    print(f"Using config: {args.config}")

    # init wandb
    wandb_run = None
    if cfg["wandb"]["enabled"]:
        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=cfg["wandb"]["run_name"],
            dir=cfg["wandb"]["dir"],
            config=cfg,
        )

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if resume_from:
        checkpoint = load_checkpoint(
            Path(resume_from),
            model,
            optimizer,
            cfg["device"],
            load_optimizer=not args.no_resume_optimizer,
            restore_rng=not args.no_restore_rng,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss)))
        print(f"Resumed from {resume_from}: next epoch={start_epoch}; global_step={global_step}; best_val_loss={best_val_loss}")

    # whatever man...
    if start_epoch > cfg["train"]["epochs"]:
        raise ValueError(f"Checkpoint resumes at epoch {start_epoch}, but train.epochs is {cfg['train']['epochs']}. Increase --epochs to continue training.")

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        device = cfg["device"]
        for frames, actions in tqdm(train_loader):
            optimizer.zero_grad()
            frames = prepare_frames(frames, device)
            actions = actions.to(device, non_blocking=True)
            use_amp = (device == "cuda" and cfg["bf16"])
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                input_frames = frames[:,:-1]
                target_frames = frames[:,1:]
                input_actions = actions[:,:-1]

                with torch.no_grad():
                    state_emb, act_emb, vis_patches = world_model.encode(input_frames, input_actions, return_patches=True)
                    pred_state_emb = world_model.predict(state_emb, act_emb)

                pred_frames = model(pred_state_emb, vis_patches, act_emb)

                l1_loss = L1(pred_frames, target_frames)
                # lpips expects inputs to be BCHW and in [0,1] range
                lpips_loss = LPIPS((pred_frames.flatten(0, 1)*2 - 1), (target_frames.flatten(0, 1)*2 - 1)).mean()
                loss = l1_loss + lpips_weight * lpips_loss

                # k step rollout
                if use_rollout:
                    rollout_weight = cfg["loss"]["rollout"]["weight"]
                    k = input_frames.shape[1]
                    history = [input_frames[:, 0:1]]
                    rollout_preds = []
                    for idx in range(1, k + 1):
                        rollout_frames = torch.cat(history, dim=1)
                        with torch.no_grad():
                            rollout_state_emb, rollout_act_emb, rollout_vis_patches = world_model.encode(rollout_frames, input_actions[:, :idx], return_patches=True)
                            rollout_pred_state_emb = world_model.predict(rollout_state_emb, rollout_act_emb)
                        rollout_vis_patches = rollout_vis_patches.reshape(
                                    input_frames.shape[0],
                                    idx,
                                    rollout_vis_patches.shape[1],
                                    rollout_vis_patches.shape[2],
                                )
                        pred_next = model(
                                    rollout_pred_state_emb[:, -1:],
                                    rollout_vis_patches[:, -1],
                                    rollout_act_emb[:, -1:],
                                )
                        history.append(pred_next)
                        rollout_preds.append(pred_next)
                    rollout_preds = torch.cat(rollout_preds, dim=1)
                    step_weights = torch.arange(1,k + 1,device=rollout_preds.device,dtype=torch.float32)
                    step_weights = step_weights / step_weights.sum()
                    per_step_l1 = (rollout_preds.float() - target_frames.float()).abs().mean(dim=(0, 2, 3, 4))
                    rollout_loss = rollout_weight*(per_step_l1*step_weights).sum()
                    loss = loss + rollout_loss
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                log = {
                    "train/loss": float(loss.detach().cpu()),
                    "train/l1_loss": float(l1_loss.detach().cpu()),
                    "train/lpips_loss": float(lpips_loss.detach().cpu()),
                    "train/rollout": float(rollout_loss.detach().cpu()) if use_rollout else 0,
                    "epoch": epoch,
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if wandb_run is not None:
                    log[f"train/E{epoch}_{global_step}.png"] = make_wandb_image(pred_frames, target_frames, n_images=2)
                    wandb_run.log(log, step=global_step)

        if (epoch % cfg["train"]["val_interval"] == 0) or epoch==cfg["train"]["epochs"]:
            model.eval()
            total_val_loss = 0
            total_val_rollout_loss = 0
            val_image = None
            with torch.no_grad():
                for frames, actions in tqdm(val_loader):
                    frames = prepare_frames(frames, device)
                    actions = actions.to(device, non_blocking=True)
                    use_amp = (device == "cuda" and cfg["bf16"])
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        input_frames = frames[:,:-1]
                        target_frames = frames[:,1:]
                        input_actions = actions[:,:-1]

                        with torch.no_grad():
                            state_emb, act_emb, vis_patches = world_model.encode(input_frames, input_actions, return_patches=True)
                            pred_state_emb = world_model.predict(state_emb, act_emb)

                        pred_frames = model(pred_state_emb, vis_patches, act_emb)
                        l1_loss = L1(pred_frames, target_frames)
                        total_val_loss += float(l1_loss.detach().cpu())
                        if val_image is None and wandb_run is not None:
                            val_image = make_wandb_image(pred_frames, target_frames, n_images=5)
                        # k step rollout
                        if use_rollout:
                            rollout_weight = cfg["loss"]["rollout"]["weight"]
                            k = input_frames.shape[1]
                            history = [input_frames[:, 0:1]]
                            rollout_preds = []
                            for idx in range(1, k + 1):
                                rollout_frames = torch.cat(history, dim=1)
                                with torch.no_grad():
                                    rollout_state_emb, rollout_act_emb, rollout_vis_patches = world_model.encode(rollout_frames, input_actions[:, :idx], return_patches=True)
                                    rollout_pred_state_emb = world_model.predict(rollout_state_emb, rollout_act_emb)
                                rollout_vis_patches = rollout_vis_patches.reshape(
                                    input_frames.shape[0],
                                    idx,
                                    rollout_vis_patches.shape[1],
                                    rollout_vis_patches.shape[2],
                                )
                                pred_next = model(
                                    rollout_pred_state_emb[:, -1:],
                                    rollout_vis_patches[:, -1],
                                    rollout_act_emb[:, -1:],
                                )
                                history.append(pred_next)
                                rollout_preds.append(pred_next)
                            rollout_preds = torch.cat(rollout_preds, dim=1)
                            step_weights = torch.arange(1,k + 1,device=rollout_preds.device,dtype=torch.float32)
                            step_weights = step_weights / step_weights.sum()
                            per_step_l1 = (rollout_preds.float() - target_frames.float()).abs().mean(dim=(0, 2, 3, 4))
                            rollout_loss = rollout_weight*(per_step_l1*step_weights).sum()
                            total_val_rollout_loss += float(rollout_loss.detach().cpu())

            if wandb_run is not None:
                val_log = {
                    "val/loss": total_val_loss/len(val_loader),
                    "val/rollout_loss": total_val_rollout_loss/len(val_loader),
                    "epoch": epoch,
                    "step": global_step,
                }
                if val_image is not None:
                    val_log[f"val/E{epoch}_{global_step}.png"] = val_image
                wandb_run.log(val_log, step=global_step)

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                save_checkpoint(out_dir/"best.pt", model, optimizer, cfg, epoch, global_step, total_val_loss, best_val_loss)
            save_checkpoint(out_dir/"latest.pt", model, optimizer, cfg, epoch, global_step, total_val_loss, best_val_loss)
            if epoch == cfg["train"]["epochs"]:
                save_checkpoint(out_dir/f"last_E{epoch}.pt", model, optimizer, cfg, epoch, global_step, total_val_loss, best_val_loss)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
