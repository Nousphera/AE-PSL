"""Real public runner wiring with bounded external acquisition fixtures."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from ae_psl.models.vision_transformer.base.vision_transformer_base import VisionTransformerBase
from ae_psl.experiment import parse_experiment
from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space


class FixtureBackbone(nn.Module):
    """Only test acquisition is replaced; real LoRA, codec and trainers run."""
    def __init__(self, classes):
        super().__init__()
        self.use_lora = True
        self.vit = VisionTransformer(image_size=16, patch_size=8, num_layers=3,
            num_heads=2, hidden_dim=8, mlp_dim=16, num_classes=classes)
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        VisionTransformerBase._apply_lora_to_encoder(self, 2, 4)
        for parameter in self.vit.heads.parameters():
            parameter.requires_grad = True
        nn.init.normal_(self.vit.heads.head.weight, std=.1)

    def get_hidden_dim(self):
        return 8


class FixtureData:
    def __init__(self):
        self.train_ds = TensorDataset(torch.randn(5, 3, 16, 16), torch.arange(5) % 3)
        self.partitioner = SimpleNamespace(_partition_id_to_indices={0:[0,1], 1:[2,3,4]},
            _determine_partition_id_to_indices_if_needed=lambda:None)

    def load_partition(self, partition_id):
        return Subset(self.train_ds, self.partitioner._partition_id_to_indices[partition_id])

    def load_validation_set(self):
        return None

    def load_test_set(self):
        return TensorDataset(torch.randn(3, 3, 16, 16), torch.arange(3))

    def get_collate_fn(self):
        return None


@pytest.mark.parametrize('personalized', [False, True])
@pytest.mark.parametrize('method', ['rand-top-k', 'c3-sl', 'adc', 'uncompressed'])
def test_public_runner_trains_and_evaluates_every_baseline(tmp_path, monkeypatch, method, personalized):
    from ae_psl import main_AE_PSL_global_evaluation as global_runner
    from ae_psl import main_AE_PSL_local_evaluation as local_runner
    from ae_psl.available_datasets import prepared
    runner = local_runner if personalized else global_runner
    dataset = 'femnist' if personalized else 'food101'
    args, _ = namespace_to_global_args_and_search_space(
        parse_experiment(['--dataset', dataset, '--method', method]))
    args.update(nr_of_epochs=1, nr_of_clients=2, batch_size=4, split_layer=1,
        save_file_name=str(tmp_path / 'experiment'), save_final_model=True,
        num_workers_finetuning=0, num_workers_activation_loading=0,
        model_weights_dir=str(tmp_path), ae_weights_dir=str(tmp_path/'ae'),
        c3_sl_compression_ratio=2, rand_top_k_sparsity=.25, adc_compression_ratio=.25,
        use_compression_during_local_eval=True, use_compression_during_global_eval=False,
        profile_flops=method == 'adc', val_mode='none', device='cpu')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'get_base_model', lambda *a, **k:FixtureBackbone(62 if personalized else 101))
    if personalized:
        batch = {0:(torch.randn(2,3,16,16),torch.tensor([0,1])),
                 1:(torch.randn(3,3,16,16),torch.tensor([0,1,2]))}
        tests = {cid:[value] for cid,value in batch.items()}
        monkeypatch.setattr(prepared, 'femnist_loaders', lambda *a:([batch],{},tests))
        runner.run_2_stage_mpsl_personalized(args, {})
    else:
        monkeypatch.setattr(runner.datasets, 'load_data', lambda **k:FixtureData())
        runner.run_2_stage_mpsl(args, {})
    result = json.loads((tmp_path/'experiment.json').read_text())
    assert result['completion']['state'] == 'completed'
    metrics = result['finetune_results']
    assert metrics['last_completed_epoch'] == 0
    assert metrics['final_test_loss'] >= 0
    assert 0 <= metrics['final_test_accuracy'] <= 1
    assert metrics['epochs'] == [0]  # No GA/CSA for heuristics or identity.
    assert metrics['outgoing_client_communication_overhead_in_mb']['0'] > 0
    snapshot = torch.load(tmp_path/'experiment.pth', weights_only=True)
    assert bool('client_models' in snapshot) == personalized
