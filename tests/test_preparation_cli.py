"""Usable preparation commands preserve data and report missing licensed inputs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from urllib.error import URLError

from PIL import Image
import pytest

from ae_psl import prepare_data
from test_preparation import image_source, leaf_source


def test_simple_cli_uses_default_raw_source_and_revalidates_on_rerun(tmp_path, capsys):
    source = image_source(tmp_path / 'raw/cifar100', per_class=2)
    args = ['--dataset', 'cifar100', '--raw-root', str(tmp_path / 'raw'),
            '--root', str(tmp_path / 'prepared'), '--allow-subset']
    assert prepare_data.main(args) == 0
    manifest = tmp_path / 'prepared/cifar100/manifest.json'
    original = manifest.read_bytes()
    assert prepare_data.main(args) == 0
    assert manifest.read_bytes() == original
    value = prepare_data.validate_manifest(manifest, True)
    assert value['provenance']['source'] == str(source)
    first = manifest.parent / value['records'][0]['path']
    first.write_bytes(b'broken')
    assert prepare_data.main(args) != 0
    assert manifest.read_bytes() == original


def test_imagenet_val_is_preserved_as_held_out_test(tmp_path):
    source = image_source(tmp_path / 'raw/imagenet100', per_class=1)
    (source / 'test').rename(source / 'val')
    args = ['--dataset', 'imagenet100', '--raw-root', str(tmp_path / 'raw'),
            '--root', str(tmp_path / 'prepared'), '--allow-subset']
    assert prepare_data.main(args) == 0
    value = prepare_data.validate_manifest(tmp_path / 'prepared/imagenet100/manifest.json', True)
    assert len([r for r in value['records'] if r['split'] == 'train']) == 3
    assert all(r['id'].startswith('val/') for r in value['records'] if r['split'] == 'test')
    assert value['provenance']['evaluation_source_split'] == 'val'


def test_simple_cli_missing_imagenet_explains_exact_layout(tmp_path, capsys):
    assert prepare_data.main(['--dataset', 'imagenet100', '--raw-root', str(tmp_path/'raw'),
                              '--root', str(tmp_path/'prepared')]) != 0
    output = capsys.readouterr()
    message = output.out + output.err
    assert 'imagenet100/train/' in message and 'imagenet100/val/' in message
    assert '--dataset imagenet100' in message
    assert not (tmp_path/'prepared/imagenet100').exists()


def test_default_all_prepares_available_sources_but_fails_for_missing_imagenet(tmp_path, capsys):
    for name in ('cifar100', 'food101', 'sun397'):
        image_source(tmp_path/'raw'/name, per_class=1)
    leaf_source(tmp_path/'raw/femnist', writers=2)
    assert prepare_data.main(['--raw-root', str(tmp_path/'raw'), '--root',
                              str(tmp_path/'prepared'), '--allow-subset']) != 0
    assert sorted(p.name for p in (tmp_path/'prepared').iterdir()) == [
        'cifar100', 'femnist', 'food101', 'sun397']
    assert 'manual input required' in capsys.readouterr().out.lower()


def test_partial_local_source_fails_without_starting_download(tmp_path, monkeypatch):
    source = image_source(tmp_path/'raw/food101', per_class=1)
    (source/'test').rename(source/'evaluation')
    monkeypatch.setattr(prepare_data, 'acquire_official', lambda *a, **k: pytest.fail('network used'))
    assert prepare_data.main(['--dataset','food101','--raw-root',str(tmp_path/'raw'),
                              '--root',str(tmp_path/'prepared'),'--allow-subset']) != 0
    assert not (tmp_path/'prepared/food101').exists()


def test_automatic_femnist_resolves_and_records_immutable_revision(tmp_path, monkeypatch):
    revision = 'b' * 40
    class Rows(list):
        _fingerprint = 'fixture'
    rows = Rows([{'image':Image.new('L',(28,28),i), 'character': i, 'writer_id': 'writer'}
                for i in range(3)])
    def load_dataset(name, **kwargs):
        assert name == 'flwrlabs/femnist' and kwargs['revision'] == revision
        assert kwargs['split'] == 'train'
        return rows
    class Hub:
        def dataset_info(self, name, revision):
            assert name == 'flwrlabs/femnist' and revision == 'main'
            return SimpleNamespace(sha='b' * 40)
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=Hub))
    assert prepare_data.main(['--dataset','femnist','--raw-root',str(tmp_path/'raw'),
                              '--root',str(tmp_path/'prepared')]) == 0
    value = prepare_data.validate_manifest(tmp_path/'prepared/femnist/manifest.json',True)
    assert value['provenance']['revision'] == revision
    assert {r['split'] for r in value['records']} == {'pool'}
    assert {r['writer'] for r in value['records']} == {'writer'}


def test_official_sun_failure_falls_back_to_checksum_verified_raw_archive(tmp_path, monkeypatch):
    import io
    import tarfile
    from torchvision.datasets import utils
    archive = tmp_path/'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('SUN397/ClassName.txt')
        member.size = 2
        stream.addfile(member,io.BytesIO(b'/a'))
    monkeypatch.setattr(prepare_data.datasets.SUN397, '_DATASET_MD5', hashlib.md5(archive.read_bytes()).hexdigest())
    def download(url, root, filename, md5):
        if 'princeton.edu' in url:
            raise URLError('host unavailable')
        (Path(root)/filename).write_bytes(archive.read_bytes())
    monkeypatch.setattr(utils,'download_url',download)
    provenance = prepare_data.acquire_official('sun397',tmp_path/'raw')
    assert 'huggingface.co/' in provenance['url']
    assert (tmp_path/'raw/SUN397/ClassName.txt').read_text() == '/a'
    assert provenance['archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_local_official_sun_tree_preserves_seed42_split_without_download(tmp_path, monkeypatch):
    source = tmp_path/'raw/sun397'
    tree = source/'SUN397'
    tree.mkdir(parents=True)
    (tree/'ClassName.txt').write_text('/a/class_a\n/b/class_b\n')
    for name in ('a/class_a','b/class_b'):
        folder = tree/name
        folder.mkdir(parents=True)
        for i in range(5):
            Image.new('RGB',(4,4),(i,0,0)).save(folder/f'sun_{i}.jpg')
    monkeypatch.setattr(prepare_data,'acquire_official',lambda *a, **k: pytest.fail('network used'))
    assert prepare_data.main(['--dataset','sun397','--raw-root',str(tmp_path/'raw'),
                              '--root',str(tmp_path/'prepared'),'--allow-subset']) == 0
    value = prepare_data.validate_manifest(tmp_path/'prepared/sun397/manifest.json',True)
    assert len([r for r in value['records'] if r['split']=='train']) == 8
    assert len([r for r in value['records'] if r['split']=='test']) == 2
    assert value['provenance']['mode'] == 'official-local'
    assert 'seed 42' in value['provenance']['split']


def test_explicit_leaf_source_can_be_reused_without_changing_provenance(tmp_path):
    source = leaf_source(tmp_path/'leaf',writers=2)
    args = ['--dataset','femnist','--source',str(source),'--root',str(tmp_path/'prepared')]
    assert prepare_data.main(args) == 0
    manifest = tmp_path/'prepared/femnist/manifest.json'
    original = manifest.read_bytes()
    assert prepare_data.main(args) == 0
    assert manifest.read_bytes() == original


def test_explicit_archive_wins_over_automatic_local_folder_discovery(tmp_path, monkeypatch):
    import tarfile
    source = tmp_path/'official/food-101'
    (source/'meta').mkdir(parents=True)
    for split in ('train','test'):
        (source/'meta'/f'{split}.json').write_text(json.dumps({c:[f'{c}/{split}'] for c in ('a','b')}))
        for c in ('a','b'):
            folder = source/'images'/c
            folder.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(4,4)).save(folder/f'{split}.jpg')
    archive = tmp_path/'food-101.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        stream.add(source,arcname='food-101')
    monkeypatch.setattr(prepare_data.datasets.Food101,'_MD5',hashlib.md5(archive.read_bytes()).hexdigest())
    image_source(tmp_path/'raw/food101',per_class=1)
    assert prepare_data.main(['--dataset','food101','--archive',str(archive),
                              '--raw-root',str(tmp_path/'raw'),'--root',str(tmp_path/'prepared'),
                              '--allow-subset']) == 0
    value = prepare_data.validate_manifest(tmp_path/'prepared/food101/manifest.json',True)
    assert value['classes'] == ['a','b']
    assert value['provenance']['archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_local_official_sun_rejects_image_symlink_escape(tmp_path):
    tree = tmp_path/'raw/sun397/SUN397'
    folder = tree/'a/class_a'
    folder.mkdir(parents=True)
    (tree/'ClassName.txt').write_text('/a/class_a\n')
    Image.new('RGB',(4,4)).save(tmp_path/'outside.jpg')
    (folder/'sun_0.jpg').symlink_to(tmp_path/'outside.jpg')
    for i in range(1,5):
        Image.new('RGB',(4,4),(i,0,0)).save(folder/f'sun_{i}.jpg')
    assert prepare_data.main(['--dataset','sun397','--raw-root',str(tmp_path/'raw'),
                              '--root',str(tmp_path/'prepared'),'--allow-subset']) != 0
    assert not (tmp_path/'prepared/sun397').exists()
