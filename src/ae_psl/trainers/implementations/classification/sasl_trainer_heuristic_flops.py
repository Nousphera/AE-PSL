"""Operator profiling of the same serialized baseline protocol.

Counts the whole local client/server run, not the paper's client-only overhead.
"""
from torch.utils.flop_counter import FlopCounterMode
from ae_psl.trainers.implementations.classification.sasl_trainer_heuristic import SASLTrainerHeuristic


class SASLTrainerHeuristicFLOPS(SASLTrainerHeuristic):
    def _single_epoch(self, *args, **kwargs):
        with FlopCounterMode(display=False) as counter:
            result = super()._single_epoch(*args, **kwargs)
        self.last_total_flops = counter.get_total_flops()
        print(f'Profiled local training operator FLOPs: {self.last_total_flops}')
        return result
