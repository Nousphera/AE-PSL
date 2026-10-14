"""Profile the corrected trainer without maintaining a divergent algorithm copy.

These are PyTorch-counted operator FLOPs for the local reference run, including
client and server work. They are not the paper's per-client overhead measurement.
"""
from torch.utils.flop_counter import FlopCounterMode
from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer


class SASLTrainerFLOPS(SASLTrainer):
    def _single_epoch(self, *args, **kwargs):
        with FlopCounterMode(display=False) as counter:
            result = super()._single_epoch(*args, **kwargs)
        self.last_total_flops = counter.get_total_flops()
        print(f"Profiled local training operator FLOPs: {self.last_total_flops}")
        return result
