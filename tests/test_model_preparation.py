"""Checkpoint preparation validates immutable bytes before exposing the torch cache."""
import hashlib
import json
from pathlib import Path

import pytest
import torch


def configure_fixture(module, monkeypatch, tmp_path):
    source = tmp_path/'provider.pth'
    torch.save({'weight': torch.arange(4)}, source)
    monkeypatch.setattr(module, 'WEIGHTS_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(module, 'WEIGHTS_SIZE', source.stat().st_size)
    return source


def test_model_preparation_publishes_torch_compatible_cache_and_reuses_valid_file(tmp_path, monkeypatch):
    from ae_psl import model_preparation
    source = configure_fixture(model_preparation, monkeypatch, tmp_path)
    torch_home = tmp_path/'torch'
    monkeypatch.setenv('TORCH_HOME', str(torch_home))
    monkeypatch.setattr(torch.hub, '_hub_dir', None)
    result = model_preparation.prepare_model(source=source)
    checkpoint = torch_home/'hub/checkpoints/vit_b_32-d86f8d99.pth'
    assert result == checkpoint
    assert torch.load(checkpoint, weights_only=True)['weight'].tolist() == [0,1,2,3]
    metadata = json.loads(checkpoint.with_suffix('.metadata.json').read_text())
    assert metadata['weights'] == 'ViT_B_32_Weights.IMAGENET1K_V1'
    assert metadata['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    timestamp = checkpoint.stat().st_mtime_ns
    assert model_preparation.prepare_model() == checkpoint
    assert checkpoint.stat().st_mtime_ns == timestamp
    checkpoint.write_bytes(b'broken')
    with pytest.raises(ValueError,match='checkpoint'):
        model_preparation.prepare_model()
    assert checkpoint.read_bytes() == b'broken'


def test_model_preparation_rejects_wrong_download_without_publishing(tmp_path, monkeypatch):
    from ae_psl import model_preparation
    configure_fixture(model_preparation, monkeypatch, tmp_path)
    def download(url, destination, **kwargs):
        Path(destination).write_bytes(b'wrong download')
    monkeypatch.setattr(torch.hub,'download_url_to_file',download)
    with pytest.raises(ValueError, match='checkpoint'):
        model_preparation.prepare_model(torch_home=tmp_path/'torch')
    assert not list((tmp_path/'torch').rglob('*.pth'))
    assert not list((tmp_path/'torch').rglob('*.metadata.json'))


def test_model_preparation_preserves_conflicting_metadata(tmp_path, monkeypatch):
    from ae_psl import model_preparation
    source = configure_fixture(model_preparation, monkeypatch, tmp_path)
    checkpoint = model_preparation.prepare_model(torch_home=tmp_path/'torch',source=source)
    metadata = checkpoint.with_suffix('.metadata.json')
    metadata.write_text('{"weights":"different"}')
    with pytest.raises(ValueError,match='metadata'):
        model_preparation.prepare_model(torch_home=tmp_path/'torch')
    assert metadata.read_text() == '{"weights":"different"}'


def test_model_cli_returns_failure_for_missing_source(tmp_path, capsys):
    from ae_psl import model_preparation
    assert model_preparation.main(['--source',str(tmp_path/'missing.pth'),
                                   '--torch-home',str(tmp_path/'torch')]) != 0
    assert 'failed' in capsys.readouterr().err.lower()
