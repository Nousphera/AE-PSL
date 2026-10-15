"""GA caches are bounded, atomic, reusable and safe for DataLoader workers."""
import pickle
import weakref

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ae_psl.ae_trainers import ae_trainer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.references = []
        self.calls = 0
        self.fail_after = None

    def get_hidden_dim(self):
        return 4

    def retrieve_split_layer_activations(self, x, split):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError('feature extraction interrupted')
        # More than one previous output retained would accumulate the corpus.
        assert sum(reference() is not None for reference in self.references) <= 1
        result = (2 * x + split).detach()
        self.references.append(weakref.ref(result))
        self.calls += 1
        return result


def data_loader(*, drop_last=False):
    values = torch.arange(60, dtype=torch.float32).reshape(5, 3, 4)
    return DataLoader(TensorDataset(values, torch.zeros(5)), batch_size=2, drop_last=drop_last)


def test_activation_cache_streams_batches_and_reads_independent_samples(tmp_path):
    model = FeatureExtractor()
    path = tmp_path/'features.npy'
    dataset = ae_trainer.extract_activations(model, 5, data_loader(), 'cpu', cache_path=path)
    assert model.calls == 3
    assert len(dataset) == 5
    expected = torch.arange(60, dtype=torch.float32).reshape(5, 3, 4) * 2 + 5
    for index in range(5):
        torch.testing.assert_close(dataset[index][0], expected[index])
    # Reading and mutating an example must not mutate the persistent cache.
    dataset[0][0].zero_()
    torch.testing.assert_close(dataset[0][0], expected[0])
    assert isinstance(np.load(path, mmap_mode='r', allow_pickle=False), np.memmap)
    assert len(pickle.dumps(dataset)) < 1024
    assert all(reference() is None for reference in model.references)


def test_interrupted_extraction_preserves_previous_cache_and_removes_partial_file(tmp_path):
    path = tmp_path/'features.npy'
    ae_trainer.extract_activations(FeatureExtractor(), 5, data_loader(), 'cpu', cache_path=path)
    before = path.read_bytes()
    failed = FeatureExtractor()
    failed.fail_after = 1
    with pytest.raises(RuntimeError, match='interrupted'):
        ae_trainer.extract_activations(failed, 5, data_loader(), 'cpu', cache_path=path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_partial_loader_does_not_publish_an_incomplete_cache(tmp_path):
    path = tmp_path/'features.npy'
    with pytest.raises(ValueError, match='samples'):
        ae_trainer.extract_activations(FeatureExtractor(), 5, data_loader(drop_last=True),
                                      'cpu', cache_path=path)
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_truncated_cache_is_rejected_before_training(tmp_path):
    path = tmp_path/'features.npy'
    ae_trainer.extract_activations(FeatureExtractor(), 5, data_loader(), 'cpu', cache_path=path)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match='cache'):
        ae_trainer.ActivationDataset(path)


def test_activation_dataset_can_be_reopened_by_a_spawn_worker(tmp_path):
    dataset = ae_trainer.extract_activations(FeatureExtractor(), 5, data_loader(), 'cpu',
                                            cache_path=tmp_path/'features.npy')
    dataset[0]  # Open a mapping in the parent before worker serialization.
    batches = list(DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context='spawn'))
    assert [len(batch[0]) for batch in batches] == [2, 2, 1]
    torch.testing.assert_close(batches[-1][0][0], torch.arange(48,60,dtype=torch.float32).reshape(3,4)*2+5)


def test_ga_reuses_disk_features_when_retraining_the_autoencoder(tmp_path, monkeypatch):
    from ae_psl.orchestrator import setup_orchestrator_parser
    from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space
    args, _ = namespace_to_global_args_and_search_space(
        setup_orchestrator_parser().parse_args(['--dataset', 'food101']))
    args.update(ae_pretrain_dataset='food101', ae_force_retrain=True, ae_pretrain_epochs=1,
        ae_pretrain_batch_size=2, ae_latent_dim=2, val_mode='none',
        num_workers_activation_loading=0, num_workers_ae_pretraining=0)
    monkeypatch.setenv('AE_WEIGHTS_DIR', str(tmp_path/'ae'))
    monkeypatch.chdir(tmp_path)

    class Data:
        train_ds = data_loader().dataset
        def load_test_set(self):
            return TensorDataset(torch.ones(2,3,4), torch.zeros(2))
        def get_collate_fn(self):
            return None

    model = FeatureExtractor()
    _, first = ae_trainer.get_auto_encoder(args, model, 'cpu', Data())
    assert first['epochs'] == [0]
    initial_calls = model.calls
    assert initial_calls > 0
    model.fail_after = initial_calls
    args.update(ae_type='1_layer_MLP_(d_d_z)_linear', ae_latent_dim=1, ae_pretrain_batch_size=1)
    _, second = ae_trainer.get_auto_encoder(args, model, 'cpu', Data())
    assert second['epochs'] == [0]
    assert model.calls == initial_calls
    assert len(list((tmp_path/'data/activations').glob('*.npy'))) == 2


def test_feature_signature_reuses_architectures_heads_and_suffix_but_binds_prefix_and_data():
    from types import SimpleNamespace
    from torchvision.models.vision_transformer import VisionTransformer
    model = SimpleNamespace(vit=VisionTransformer(image_size=16, patch_size=8, num_layers=2,
        num_heads=2, hidden_dim=8, mlp_dim=16, num_classes=3))
    args = dict(ae_pretrain_dataset='imagenet', ae_pretrain_dataset_fraction=1.,
        model='vit_b_32', split_layer=1, random_seed=42, num_workers_activation_loading=0,
        ae_latent_dim=64, ae_type='2_layer_MLP_(d_d_d_z)', ae_pretrain_epochs=30,
        ae_pretrain_batch_size=512, small_test_run=False)
    dataset = data_loader().dataset
    original = ae_trainer.feature_cache_signature(args, model, dataset, dataset, 'source-a')
    with torch.no_grad():
        model.vit.heads.head.weight.add_(1)
        model.vit.encoder.layers[1].ln_1.weight.add_(1)
    changed = {**args, 'ae_latent_dim':24, 'ae_type':'1_layer_MLP_(d_d_z)',
               'ae_pretrain_epochs':2, 'ae_pretrain_batch_size':16}
    assert ae_trainer.feature_cache_signature(changed, model, dataset, dataset, 'source-a') == original
    assert ae_trainer.feature_cache_signature(args, model, dataset, dataset, 'source-b') != original
    with torch.no_grad():
        model.vit.encoder.layers[0].ln_1.weight.add_(1)
    assert ae_trainer.feature_cache_signature(args, model, dataset, dataset, 'source-a') != original


def test_ga_cache_hit_and_miss_leave_identical_caller_random_streams(tmp_path, monkeypatch):
    import random
    from ae_psl.orchestrator import setup_orchestrator_parser
    from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space
    args, _ = namespace_to_global_args_and_search_space(
        setup_orchestrator_parser().parse_args(['--dataset', 'food101']))
    args.update(ae_pretrain_dataset='food101', ae_force_retrain=False, ae_pretrain_epochs=1,
        ae_pretrain_batch_size=2, ae_latent_dim=2, val_mode='none',
        num_workers_activation_loading=0, num_workers_ae_pretraining=0)
    monkeypatch.setenv('AE_WEIGHTS_DIR', str(tmp_path/'ae'))
    monkeypatch.chdir(tmp_path)

    class Data:
        train_ds = TensorDataset(torch.ones(2,3,4), torch.zeros(2))
        def load_test_set(self):
            return self.train_ds
        def get_collate_fn(self):
            return None

    model = FeatureExtractor()
    samples = []
    for _ in range(2):
        torch.manual_seed(999)
        np.random.seed(99)
        random.seed(9)
        _, results = ae_trainer.get_auto_encoder(args, model, 'cpu', Data())
        samples.append((torch.rand(3), np.random.rand(3), random.random()))
    assert results['reused_AE']
    torch.testing.assert_close(samples[0][0], samples[1][0], rtol=0, atol=0)
    np.testing.assert_array_equal(samples[0][1], samples[1][1])
    assert samples[0][2] == samples[1][2]
    torch.manual_seed(999)
    torch.testing.assert_close(samples[0][0], torch.rand(3), rtol=0, atol=0)


def test_cache_checks_free_space_before_mapping_the_full_shape(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: shutil._ntuple_diskusage(100, 100, 0))
    with pytest.raises(OSError, match='space'):
        ae_trainer.extract_activations(FeatureExtractor(), 5, data_loader(), 'cpu',
                                      cache_path=tmp_path/'features.npy')
    assert not list(tmp_path.iterdir())
