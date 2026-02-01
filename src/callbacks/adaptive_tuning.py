import pytorch_lightning as pl
import torch
import numpy as np

class AdaptiveCriticTuning(pl.Callback):
    def __init__(self, w_low=-5.0, ma_window=20):
        super().__init__()
        self.w_low = w_low
        self.w_gap_history = []
        self.ma_window = ma_window

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        w_gap = trainer.callback_metrics.get("w_gap")
        if w_gap is None:
            return

        w_gap = w_gap.item() if isinstance(w_gap, torch.Tensor) else w_gap
        
        self.w_gap_history.append(w_gap)
        if len(self.w_gap_history) > self.ma_window:
            self.w_gap_history.pop(0)
        
        if len(self.w_gap_history) > 0:
            w_gap_ma = np.mean(self.w_gap_history)
            pl_module.log("train/w_gap_ma_500", w_gap_ma)
            
            # UNIDIRECTIONAL: Only weaken critic, never strengthen
            if w_gap_ma < self.w_low:
                 pl_module.critic_steps = max(1, pl_module.critic_steps - 1)
                 pl_module.lambda_gp = min(20.0, pl_module.lambda_gp * 1.5)
                 
                 pl_module.log("adapt/critic_steps", float(pl_module.critic_steps))
                 pl_module.log("adapt/lambda_gp", pl_module.lambda_gp)
