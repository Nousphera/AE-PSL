"""Run AE-PSL and its paper baselines with shared experiment settings."""

import json
import math
import sys

from ae_psl.orchestrator import run_orchestrator, setup_orchestrator_parser
from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space

METHODS = {
    'ae-psl': 'ae',
    'uncompressed': 'none',
    'rand-top-k': 'rand_top_k',
    'adc': 'adc',
    'c3-sl': 'c3_sl',
}


def _values(value):
    return value if isinstance(value, list) else [value]


def parse_experiment(argv=None):
    parser = setup_orchestrator_parser()
    parser.description = 'Run AE-PSL or a comparison method with the paper ViT-B/32.'
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--dirichlet_alpha', nargs='+', type=float,
                        help='Class concentration across clients; paper default: 0.5.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the effective configuration without acquiring data or training.')
    parser.set_defaults(ae_latent_dim=64, bit_width=32, rand_top_k_sparsity=0.065,
                        adc_compression_ratio=0.1, c3_sl_compression_ratio=12,
                        save_final_model=True, experiments_dir='results',
                        experiment_name=None, num_workers_finetuning=0,
                        num_workers_ae_pretraining=0, num_workers_activation_loading=0,
                        test_num_workers=0)
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(tokens)
    if args.method:
        explicit_compressor = any(x.split('=')[0] == '--compression_method' for x in tokens)
        if explicit_compressor and _values(args.compression_method) != [METHODS[args.method]]:
            parser.error('--method and --compression_method select different methods')
        args.compression_method = METHODS[args.method]
    args.experiment_name = args.experiment_name or args.method or 'ae-psl'
    if args.dirichlet_alpha is not None:
        if any(not math.isfinite(x) or x <= 0 for x in args.dirichlet_alpha):
            parser.error('--dirichlet_alpha must be finite and positive')
        args.dataset_split_type = [str(x) for x in args.dirichlet_alpha]
    del args.dirichlet_alpha
    del args.method

    for key in ('nr_of_clients', 'batch_size', 'nr_of_epochs', 'ae_pretrain_batch_size',
                'lora_rank', 'lora_alpha', 'c3_sl_compression_ratio'):
        if any(x <= 0 for x in _values(getattr(args, key))):
            parser.error(f'--{key} must be positive')
    if any(b % n for b in _values(args.batch_size) for n in _values(args.nr_of_clients)):
        parser.error('--batch_size must be divisible by every --nr_of_clients value')
    if any(not 0 <= x <= 12 for x in _values(args.split_layer)):
        parser.error('--split_layer must be between 0 and 12 for ViT-B/32')
    if any(not 1 <= x <= 768 for x in _values(args.ae_latent_dim)):
        parser.error('--ae_latent_dim must be between 1 and 768')
    for key in ('start_lr', 'ae_pretrain_start_lr', 'ae_finetune_lr'):
        if any(not math.isfinite(x) or x <= 0 for x in _values(getattr(args, key))):
            parser.error(f'--{key} must be finite and positive')
    for key in ('rand_top_k_sparsity', 'adc_compression_ratio'):
        if any(not math.isfinite(x) or not 0 < x <= 1 for x in _values(getattr(args, key))):
            parser.error(f'--{key} must be in (0, 1]')
    if not math.isfinite(args.femnist_fraction) or not 0 < args.femnist_fraction <= 1:
        parser.error('--femnist_fraction must be in (0, 1]')
    if not 0 <= args.val_split < 1:
        parser.error('--val_split must be in [0, 1)')
    for key in ('num_workers_finetuning', 'num_workers_ae_pretraining',
                'num_workers_activation_loading', 'test_num_workers',
                'ae_pretrain_epochs', 'iterative_client_specific_alignment_epochs'):
        if any(x < 0 for x in _values(getattr(args, key))):
            parser.error(f'--{key} cannot be negative')
    return args


def main(argv=None):
    args = parse_experiment(argv)
    dry_run = args.dry_run
    del args.dry_run
    if dry_run:
        config, sweep = namespace_to_global_args_and_search_space(args)
        print(json.dumps({'config': config, 'sweep': sweep}, indent=2))
        return
    run_orchestrator(args)


if __name__ == '__main__':
    main()
