import json
from pathlib import Path

import pytest
import torch


def test_method_entrypoint_selects_paper_defaults_without_running(capsys, tmp_path):
    from ae_psl.experiment import main
    main(['--dataset', 'cifar100', '--method', 'ae-psl', '--dry-run',
          '--experiments_dir', str(tmp_path / 'results')])
    record = json.loads(capsys.readouterr().out)
    assert record['config']['compression_method'] == 'ae'
    assert record['config']['ae_latent_dim'] == 64
    assert record['config']['ae_pretrain_epochs'] == 30
    assert record['config']['ae_pretrain_start_lr'] == 1e-4
    assert record['config']['ae_pretrain_cosine_eta_min'] == 5e-5
    assert record['config']['save_final_model'] is True
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('method,compression,key,value', [
    ('uncompressed', 'none', 'bit_width', 32),
    ('rand-top-k', 'rand_top_k', 'rand_top_k_sparsity', 0.065),
    ('adc', 'adc', 'adc_compression_ratio', 0.1),
    ('c3-sl', 'c3_sl', 'c3_sl_compression_ratio', 12),
])
def test_method_entrypoint_preserves_baseline_settings(method, compression, key, value):
    from ae_psl.experiment import parse_experiment
    args = parse_experiment(['--dataset', 'food101', '--method', method])
    assert args.compression_method == compression
    assert getattr(args, key) == value
    assert args.experiment_name == method


def test_method_entrypoint_accepts_sweeps_and_explicit_parameters():
    from ae_psl.experiment import parse_experiment
    args = parse_experiment(['--dataset', 'cifar100', 'food101', '--method', 'ae-psl',
                             '--ae_latent_dim', '24', '140', '--random_seed', '42', '43',
                             '--dirichlet_alpha', '0.3'])
    assert args.ae_latent_dim == [24, 140]
    assert args.random_seed == [42, 43]
    assert args.dataset_split_type == ['0.3']


@pytest.mark.parametrize('options', [
    ['--nr_of_clients', '0'], ['--batch_size', '1', '--nr_of_clients', '5'],
    ['--ae_latent_dim', '0'], ['--dirichlet_alpha', '-1'],
    ['--split_layer', '13'], ['--nr_of_epochs', '0'],
    ['--rand_top_k_sparsity', '1.2'], ['--adc_compression_ratio', '0'],
])
def test_invalid_experiment_rejected_before_acquisition(options):
    from ae_psl.experiment import parse_experiment
    with pytest.raises(SystemExit):
        parse_experiment(['--dataset', 'cifar100', *options])


def test_device_selection_honors_cpu_with_cuda_available(monkeypatch):
    from ae_psl.utils.cuda_utils import get_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert str(get_device({'device': 'cpu', 'gpu_id': None})) == 'cpu'


def test_unavailable_requested_accelerator_does_not_silently_fallback(monkeypatch):
    from ae_psl.utils.cuda_utils import get_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        get_device({'device': 'cuda', 'gpu_id': None})


def test_new_campaigns_do_not_overwrite_same_name(tmp_path):
    from ae_psl.orchestrator import generate_manifest
    a, _ = generate_manifest('same', str(tmp_path), {'marker': 1}, {})
    b, _ = generate_manifest('same', str(tmp_path), {'marker': 2}, {})
    assert a != b
    assert json.loads((Path(a) / 'manifest.json').read_text())['global_args']['marker'] == 1


def test_conflicting_method_options_are_rejected():
    from ae_psl.experiment import parse_experiment
    with pytest.raises(SystemExit):
        parse_experiment(['--dataset', 'cifar100', '--method', 'adc',
                          '--compression_method', 'ae'])


def test_legacy_compressor_sweep_is_preserved():
    from ae_psl.experiment import parse_experiment
    args = parse_experiment(['--dataset', 'cifar100', '--compression_method', 'ae', 'none'])
    assert args.compression_method == ['ae', 'none']


@pytest.mark.parametrize('options', [
    ['--scheduler', 'step'], ['--ae_pretrain_optimizer', 'adamw'],
    ['--ae_pretrain_optimizer', 'sgd'], ['--ae_pretrain_loss_fn', 'l1'],
])
def test_unimplemented_legacy_settings_are_rejected(options):
    from ae_psl.experiment import parse_experiment
    with pytest.raises(SystemExit):
        parse_experiment(['--dataset', 'cifar100', *options])


@pytest.mark.parametrize('dataset', ['cifar100', 'femnist'])
def test_profiled_campaign_keeps_settings_and_runs_every_job(tmp_path, monkeypatch, dataset):
    from ae_psl import orchestrator
    from ae_psl.experiment import main
    seen = []

    def run(config, sweep):
        seen.append(dict(config))
        assert config['nr_of_clients'] == 5
        assert config['batch_size'] == 125
        assert config['nr_of_epochs'] == 2
        assert config['profile_flops'] is True
        result = {'completion': {'state': 'completed', 'stage': 'dft',
                  'requested_epochs': 2, 'last_completed_epoch': 1},
                  'finetune_results': {'last_completed_epoch': 1,
                  'final_test_loss': 1.0, 'final_test_accuracy': 0.5}}
        Path(config['save_file_name'] + '.json').write_text(json.dumps(result))

    name = 'run_2_stage_mpsl_personalized' if dataset == 'femnist' else 'run_2_stage_mpsl'
    monkeypatch.setattr(orchestrator, name, run)
    main(['--dataset', dataset, '--method', 'uncompressed', '--profile_flops', 'True',
          '--random_seed', '42', '43', '--nr_of_epochs', '2',
          '--experiments_dir', str(tmp_path)])
    assert [item['random_seed'] for item in seen] == [42, 43]
