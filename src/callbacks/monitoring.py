import pytorch_lightning as pl
import torch
import time
import os

class PerformanceMonitoringCallback(pl.Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
        self.epoch_start_time = None
        self.batch_start_time = None
        self.last_batch_end_time = None
        self.total_muons = 0
        self.total_events = 0
        self.batches_seen = 0
        self.load_time_sum = 0
        self.step_time_sum = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        self.last_batch_end_time = self.epoch_start_time
        self.total_muons = 0
        self.total_events = 0
        self.batches_seen = 0
        self.load_time_sum = 0
        self.step_time_sum = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        curr_time = time.perf_counter()
        if self.last_batch_end_time is not None:
            self.load_time_sum += (curr_time - self.last_batch_end_time)
        self.batch_start_time = curr_time
        if self.epoch_start_time is None:
            self.epoch_start_time = self.batch_start_time

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end_time = time.perf_counter()
        self.last_batch_end_time = end_time
        step_time = end_time - self.batch_start_time
        self.step_time_sum += step_time
        
        real_muons, _, _, counts = batch
        self.total_muons += int(counts.sum().item())
        self.total_events += int(counts.size(0))
        self.batches_seen += 1

        if self.batches_seen % self.log_interval == 0:
            elapsed = end_time - self.epoch_start_time
            
            # Rate metrics
            pl_module.log("perf/batch_per_s", self.batches_seen / max(1e-9, elapsed))
            pl_module.log("perf/events_per_s", self.total_events / max(1e-9, elapsed))
            pl_module.log("perf/muons_per_s", self.total_muons / max(1e-9, elapsed))
            
            # Data stats
            pl_module.log("data/events_seen", float(self.total_events))
            pl_module.log("data/muons_seen", float(self.total_muons))
            pl_module.log("data/mean_muons_per_event", self.total_muons / max(1, self.total_events))
            
            # Per-batch stats (parity with legacy)
            pl_module.log("data/mean_counts", counts.float().mean())
            pl_module.log("data/max_counts", counts.float().max().item())
            
            # Timing
            pl_module.log("perf/avg_step_ms", (self.step_time_sum / self.batches_seen) * 1000)
            pl_module.log("perf/avg_load_ms", (self.load_time_sum / self.batches_seen) * 1000)

            # GPU Stats
            if torch.cuda.is_available():
                stats = torch.cuda.memory_stats()
                # Conversion to MiB
                mib = 1024 * 1024
                pl_module.log("cuda/alloc_mib", torch.cuda.memory_allocated() / mib)
                pl_module.log("cuda/reserved_mib", torch.cuda.memory_reserved() / mib)
                pl_module.log("cuda/max_alloc_mib", torch.cuda.max_memory_allocated() / mib)
                pl_module.log("cuda/max_reserved_mib", torch.cuda.max_memory_reserved() / mib)
                
                # Active/Inactive (requires more detailed stats if parity is strict)
                # But these 4 are the most important ones.

class HistogramLoggingCallback(pl.Callback):
    def __init__(self, log_every_n_steps=1000, max_muons=20000):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_muons = max_muons

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            real_muons, _, prims, counts = batch
            
            # Use the module's normalization if available
            if hasattr(pl_module, 'normalizer') and pl_module.normalizer:
                # We assume the user might want to see histograms of normalized data
                # parity with train.py which logs normalized histograms.
                pass
            
            # Log Real Histograms
            # The model already has conditions and real_muons_feats logic
            # but we can just use the batch here if we know the slicing.
            # However, it's safer to let the model expose them.
            if hasattr(pl_module, 'last_real_feats') and pl_module.last_real_feats is not None:
                real_cpu = pl_module.last_real_feats[:self.max_muons].detach().cpu()
                for d in range(min(real_cpu.shape[1], 8)):
                    trainer.logger.experiment.add_histogram(
                        f"real/muon_feat{d}", real_cpu[:, d], trainer.global_step
                    )

            if hasattr(pl_module, 'last_fake_feats') and pl_module.last_fake_feats is not None:
                fake_cpu = pl_module.last_fake_feats[:self.max_muons].detach().cpu()
                for d in range(min(fake_cpu.shape[1], 8)):
                    trainer.logger.experiment.add_histogram(
                        f"fake/muon_feat{d}", fake_cpu[:, d], trainer.global_step
                    )
            
            # Data counts
            trainer.logger.experiment.add_histogram(
                "data/counts", counts.detach().cpu().float(), trainer.global_step
            )
