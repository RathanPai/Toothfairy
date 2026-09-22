import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.losses import CompoundTopologyLoss
from src.evaluation.metrics import evaluate_case
from src.postprocessing.anatomical_filter import filter_bilateral_ian

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loss_fn: nn.Module = None,
        device: str = "cuda",
        grad_accum_steps: int = 2,
        checkpoint_dir: str | Path = "checkpoints",
        experiment_name: str = "teacher_fold0"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or CompoundTopologyLoss()
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_val_dice = 0.0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        running_ce, running_dice, running_cldice, running_boundary = 0.0, 0.0, 0.0, 0.0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, (images, targets, case_ids) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                
                # Main output gets full compound loss
                if isinstance(outputs, list):
                    loss_dict = self.loss_fn(outputs[0], targets)
                    main_loss = loss_dict["loss"]
                    # Deep supervision outputs use lightweight CE + Dice (saving immense VRAM)
                    ds_loss = sum(
                        (self.loss_fn.ce_loss(out, targets.long()) + self.loss_fn.dice_loss(torch.softmax(out, dim=1), targets.long())) * (0.25 ** (i + 1))
                        for i, out in enumerate(outputs[1:])
                    )
                    loss = (main_loss + ds_loss) / self.grad_accum_steps
                else:
                    loss_dict = self.loss_fn(outputs, targets)
                    loss = loss_dict["loss"] / self.grad_accum_steps
                    
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.grad_accum_steps
            running_ce += loss_dict.get("l_ce", 0)
            running_dice += loss_dict.get("l_dice", 0)
            running_cldice += loss_dict.get("l_cldice", 0)
            running_boundary += loss_dict.get("l_boundary", 0)
            
            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})
            
        num_steps = max(1, len(self.train_loader))
        return {
            "loss": total_loss / num_steps,
            "ce": running_ce / num_steps,
            "dice": running_dice / num_steps,
            "cldice": running_cldice / num_steps,
            "boundary": running_boundary / num_steps
        }

    @torch.no_grad()
    def sliding_window_inference(self, volume: torch.Tensor, patch_size: tuple = (64, 128, 128), overlap: float = 0.5) -> np.ndarray:
        """
        Memory-safe sliding-window 3D patch inference with CPU probability accumulation.
        volume: (1, 1, D, H, W) on CPU or GPU
        """
        self.model.eval()
        volume_np = volume[0, 0].cpu().numpy()
        D, H, W = volume_np.shape
        pD, pH, pW = patch_size
        num_classes = getattr(self.model, "num_classes", 3)
        
        output_probs = np.zeros((num_classes, D, H, W), dtype=np.float32)
        count_map = np.zeros((1, D, H, W), dtype=np.float32)
        
        step_d = max(1, int(pD * (1.0 - overlap)))
        step_h = max(1, int(pH * (1.0 - overlap)))
        step_w = max(1, int(pW * (1.0 - overlap)))
        
        d_starts = list(range(0, max(1, D - pD + 1), step_d))
        if d_starts[-1] != max(0, D - pD): d_starts.append(max(0, D - pD))
        h_starts = list(range(0, max(1, H - pH + 1), step_h))
        if h_starts[-1] != max(0, H - pH): h_starts.append(max(0, H - pH))
        w_starts = list(range(0, max(1, W - pW + 1), step_w))
        if w_starts[-1] != max(0, W - pW): w_starts.append(max(0, W - pW))
        
        for z in d_starts:
            for y in h_starts:
                for x in w_starts:
                    patch = volume_np[z:z+pD, y:y+pH, x:x+pW]
                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                    with torch.amp.autocast('cuda'):
                        patch_logits = self.model(patch_t)
                        if isinstance(patch_logits, list):
                            patch_logits = patch_logits[0]
                        patch_probs = torch.softmax(patch_logits, dim=1)[0].cpu().numpy()
                        
                    output_probs[:, z:z+pD, y:y+pH, x:x+pW] += patch_probs
                    count_map[:, z:z+pD, y:y+pH, x:x+pW] += 1.0
                    
        return output_probs / np.maximum(count_map, 1e-5)

    @torch.no_grad()
    def validate(self, patch_size: tuple = (64, 128, 128)) -> dict:
        self.model.eval()
        dices, hd95s = [], []
        
        for images, targets, case_ids in self.val_loader:
            case_id = case_ids[0]
            # images: (1, 1, D, H, W), targets: (1, D, H, W)
            prob_map = self.sliding_window_inference(images, patch_size=patch_size)
            pred_classes = np.argmax(prob_map, axis=0)
            target_np = targets[0].cpu().numpy()
            
            # Post-processing
            cleaned_pred = filter_bilateral_ian(pred_classes)
            
            # Metrics
            metrics = evaluate_case(cleaned_pred, target_np)
            dices.append(metrics["dice"])
            hd95s.append(metrics["hd95"])
            
        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_hd95 = float(np.mean(hd95s)) if hd95s else 100.0
        
        return {
            "val_dice": mean_dice,
            "val_hd95": mean_hd95,
            "raw_dices": dices,
            "raw_hd95s": hd95s
        }

    def train(self, max_epochs: int, save_interval: int = 25):
        print(f"\n🚀 Starting Training: {self.experiment_name} ({max_epochs} epochs)")
        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Periodic validation
            if epoch % 5 == 0 or epoch == max_epochs:
                val_metrics = self.validate()
                print(f"Epoch {epoch:03d}/{max_epochs:03d} | Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val DSC: {val_metrics['val_dice']:.4f} | Val HD95: {val_metrics['val_hd95']:.3f} mm")
                
                # Check for best model
                if val_metrics["val_dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["val_dice"]
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_dice": self.best_val_dice,
                        "val_hd95": val_metrics["val_hd95"]
                    }, self.checkpoint_dir / "best_model.pth")
                    print(f"  ⭐ New Best Model Saved! DSC: {self.best_val_dice:.4f}, HD95: {val_metrics['val_hd95']:.3f} mm")
                    
            if epoch % save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                }, self.checkpoint_dir / f"epoch_{epoch:03d}.pth")
