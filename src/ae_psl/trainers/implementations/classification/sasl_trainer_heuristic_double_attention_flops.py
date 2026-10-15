"""ADC profiling delegates to the common serialized trainer."""
from ae_psl.trainers.implementations.classification.sasl_trainer_heuristic_flops import SASLTrainerHeuristicFLOPS


class SASLTrainerHeuristicDAFLOPS(SASLTrainerHeuristicFLOPS):
    pass
