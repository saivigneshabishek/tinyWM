from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import TinyWMDataset
from model import TinyWorldModel

class SIGReg(torch.nn.Module):
    ''' <https://github.com/lucas-maes/le-wm/blob/main/module.py#L10> '''
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024, lambd=0.09):
        super().__init__()
        self.num_proj = num_proj
        self.lambd = lambd
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        returns with lambda applied
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return self.lambd*(statistic.mean()) # average over projections and time

def load_config(path):
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)

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
    return DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=train,
        num_workers=loader_cfg["num_workers"],
        pin_memory=(device == "cuda"),
        drop_last=train,
    )


def save_checkpoint(path, model, optimizer, cfg, epoch, global_step, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
        },
        path,
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/wm_A100.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])

    out_dir = Path(cfg["checkpoint"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(cfg, train=True)
    val_loader = get_dataloader(cfg, train=False)

    model = TinyWorldModel(cfg["TinyWorldModel"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=cfg["optimizer"]["lr"],
                                weight_decay=cfg["optimizer"]["weight_decay"])
    
    MSE = torch.nn.MSELoss()
    SigReg = SIGReg(knots=cfg["loss"]["sigreg"]["knots"],
                    num_proj=cfg["loss"]["sigreg"]["num_proj"],
                    lambd=cfg["loss"]["sigreg"]["lambd"]).to(device=cfg["device"])
    
    print(f"Using config: {args.config}")

    # init wandb
    wandb_run = None
    if cfg["wandb"]["enabled"]:
        import wandb
        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=cfg["wandb"]["run_name"],
            dir=cfg["wandb"]["dir"],
            config=cfg,
        )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        device = cfg["device"]
        for frames, actions in tqdm(train_loader):
            optimizer.zero_grad()
            frames = frames.to(device)
            actions = actions.to(device)
            use_amp = (device == "cuda" and cfg["bf16"])
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                state_emb, act_emb = model.encode(frames, actions)
                input_state_emb = state_emb[:,:-1]
                input_act_emb = act_emb[:,:-1]
                target_state_emb = state_emb[:,1:]
                pred_state_emb = model.predict(input_state_emb, input_act_emb)
                
                mse_loss = MSE(pred_state_emb, target_state_emb)
                loss = mse_loss + SigReg(state_emb.transpose(0,1)) # lambda is already applied!

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                log = {
                    "train/loss": float(loss.detach().cpu()),
                    "epoch": epoch,
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if wandb_run is not None:
                    wandb_run.log(log, step=global_step)

        if (epoch % cfg["train"]["val_interval"] == 0) or epoch==cfg["train"]["epochs"]:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for frames, actions in tqdm(val_loader):
                    frames = frames.to(device)
                    actions = actions.to(device)
                    use_amp = (device == "cuda" and cfg["bf16"])
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        state_emb, act_emb = model.encode(frames, actions)
                        input_state_emb = state_emb[:,:-1]
                        input_act_emb = act_emb[:,:-1]
                        target_state_emb = state_emb[:,1:]
                        pred_state_emb = model.predict(input_state_emb, input_act_emb)
                        loss = MSE(pred_state_emb, target_state_emb)
                        total_val_loss += float(loss.detach().cpu())

            if wandb_run is not None:
                val_log = {
                    "val/loss": total_val_loss/len(val_loader),
                }
                wandb_run.log(val_log, step=epoch)

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                save_checkpoint(out_dir/"best.pt", model, optimizer, cfg, epoch, global_step, total_val_loss)
            elif epoch == cfg["train"]["epochs"]:
                save_checkpoint(out_dir/f"last_E{epoch}.pt", model, optimizer, cfg, epoch, global_step, total_val_loss)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
