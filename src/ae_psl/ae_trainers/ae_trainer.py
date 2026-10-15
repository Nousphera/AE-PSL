import os
import time
import random

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from ae_psl import available_datasets

from ae_psl.ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from ae_psl.ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from ae_psl.available_datasets import DistributedDataset
from ae_psl.models.auto_encoder import initialize_AE, BaseViTAE
from ae_psl.utils.ae_registry_utils import load_auto_encoder_model, prepare_ae_dir, filename_from_signature, \
    save_ae_with_signature
from ae_psl.utils.dataloader_utils import get_dataloaders_from_datasets_AE
from ae_psl.utils.model_saving_utils import save_ae_experiment_results
from ae_psl.utils.scheduler_utils import get_ae_pretrain_optimizer_and_scheduler


from ae_psl.ae_trainers.activation_cache import (
    ActivationDataset, EXTRACTION_BATCH_SIZE, feature_cache_signature,
    feature_cache_name, feature_extractor_fingerprint, write_activation_cache,
)

# Makes the choice to pretrain or load an AE model based on global_args
def get_auto_encoder(global_args, base_model, device, full_dataset: DistributedDataset = None):
    """Isolate GA randomness so cache hits cannot change subsequent CSA/DFT."""
    target = torch.device(device)
    cuda_devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    mps_state = torch.mps.get_rng_state() if target.type == 'mps' else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            seed = global_args['random_seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            if mps_state is not None:
                torch.mps.manual_seed(seed)
            return _get_auto_encoder(global_args, base_model, device, full_dataset)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _get_auto_encoder(global_args, base_model, device, full_dataset=None):

    from ae_psl.prepare_data import sha256
    dataset_name = global_args['ae_pretrain_dataset']
    dataset_name = 'imagenet100' if dataset_name == 'imagenet' else dataset_name
    manifest = Path(os.environ.get('TORCH_DATA_DIR', 'data/datasets')) / dataset_name / 'manifest.json'
    source_digest = sha256(manifest) if manifest.exists() else None

    # This signature is used to identify the AE model. It will be used to find an existing AE model or to save a new one.
    ae_signature = {
        'type': global_args['ae_type'],
        'dataset': global_args['ae_pretrain_dataset'],
        'dataset_proportion': global_args['ae_pretrain_dataset_fraction'],
        'model': global_args['model'],
        'split_layer': global_args['split_layer'],
        'input_dim': base_model.get_hidden_dim(),
        'latent_dim': global_args['ae_latent_dim'],
        'seed': global_args['random_seed'],
        'training': {key: value for key, value in global_args.items()
                     if key.startswith('ae_pretrain_') or key in ('val_mode','val_split','small_test_run')},
        'backbone_sha256': feature_extractor_fingerprint(base_model, global_args['split_layer']),
        'dataset_root': os.path.abspath(os.environ.get('TORCH_DATA_DIR', 'data/datasets')),
        'cache_schema': 6,
        'prepared_manifest_sha256': source_digest,
        'supplied_split_sha256': getattr(full_dataset, 'source_digest', None),
        'uses_supplied_train_split': full_dataset is not None,

    }

    # Initialize an untrained AE model
    # if latent dim == hidden dim of model, then it will initialize as an identity bottleneck regardless of model type
    auto_encoder_model = initialize_AE(global_args, base_model.get_hidden_dim())
    auto_encoder_model.to(device)

    if type(auto_encoder_model) is BaseViTAE:
        global_args['ae_type'] = 'identity'  # Also skip CSA for latent_dim == hidden_dim.

    if not global_args['ae_general_alignment']:
        return auto_encoder_model, {}

    if type(auto_encoder_model) is BaseViTAE:
        print("AE type is BaseViTAE, skipping AE pretraining/loading.")
        return auto_encoder_model, {}


    if not global_args['ae_force_retrain']:
        # Load an existing AE model based on required ae signature
        model, results = load_auto_encoder_model(global_args, auto_encoder_model, ae_signature, device)
        if model is not None:
            return model, results
        else:
            print("No existing AE model found, proceeding to pretrain a new model.")
    else:
        print("AE force retrain flag is set, proceeding to pretrain a new model.")


    # If the AE pretrain dataset differs, then we are training on downstream client data
    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    if is_train_ae_on_downstream_data:
        print(f"Note: AE pre-training on downstream client dataset: {global_args['dataset']}")
    else:
        print(f"Note: AE pre-training on separate pre-train dataset: {global_args['ae_pretrain_dataset']}")


    if full_dataset is None:
        full_dataset = available_datasets.load_data(name=global_args['ae_pretrain_dataset'], num_partitions=1,
                                                    split='iid', seed=global_args['random_seed'],
                                                    global_args=global_args)


    # A supplied downstream dataset already excludes its validation subset.
    # Use every client's training examples, not only partition zero.
    train_dataset = full_dataset.train_ds if hasattr(full_dataset, 'train_ds') else full_dataset.load_partition(partition_id=0)
    test_dataset = full_dataset.load_test_set()

    if global_args['small_test_run']:
        train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 100))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 100))
    else:
        # Take the correct proportion of the dataset, as we want to experiment with how much data is needed to pretrain the AE
        train_dataset = available_datasets.Subset(train_dataset, range(0, int(len(train_dataset) * global_args[
            'ae_pretrain_dataset_fraction'])))
        test_dataset = available_datasets.Subset(test_dataset, range(0, int(len(test_dataset) * global_args[
            'ae_pretrain_dataset_fraction'])))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=EXTRACTION_BATCH_SIZE, shuffle=True,
                                  generator=torch.Generator().manual_seed(global_args['random_seed']),
                                  pin_memory=True,
                                  num_workers=global_args['num_workers_activation_loading'],
                                  collate_fn=full_dataset.get_collate_fn())

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=EXTRACTION_BATCH_SIZE, shuffle=False,
                                 generator=torch.Generator().manual_seed(global_args['random_seed'] + 1),
                                 pin_memory=True,
                                 num_workers=global_args['num_workers_activation_loading'],
                                 collate_fn=full_dataset.get_collate_fn())

    start_time = time.time()


    # Ensure activation cache directory exists and try loading cached activations first.
    activations_dir = os.path.join('data', 'activations')
    os.makedirs(activations_dir, exist_ok=True)

    signature = feature_cache_signature(global_args, base_model, train_dataset, test_dataset,
                                        source_digest or getattr(full_dataset, 'source_digest', None))
    base_name = feature_cache_name(signature)
    cache_datasets = []
    for offset, (split, loader) in enumerate((('train', train_dataloader), ('test', test_dataloader))):
        path = Path(activations_dir) / f'{base_name}_{split}.npy'
        if path.exists():
            print(f'Opening frozen-feature cache {path}')
            dataset = ActivationDataset(path, expected_count=len(loader.dataset))
        else:
            print(f'Extracting frozen-feature cache {path}')
            dataset = extract_activations(base_model, global_args['split_layer'], loader, device,
                global_args['ae_pretrain_dataset'] == 'cifar100', cache_path=path,
                seed=global_args['random_seed'] + offset)
        cache_datasets.append(dataset)
    train_acts_dataset, test_acts_dataset = cache_datasets

    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time

    print(f"Activation extraction completed in: {duration:.2f} seconds")

    return pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset, ae_signature, device)




def pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset,
                          ae_signature, device,
                          early_stopping_patience=3):
    """
    Args:
        validation_mode (str): Options are 'none', 'early_saving', 'early_stopping'.
        model_save_path (str): Path to save the best model checkpoint.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    # ga validation_mode
    # val_split

    validation_mode = global_args['val_mode']
    validation_split = global_args['val_split']
    if validation_split == 0: validation_mode = 'none'

    train_acts_dataloader, val_acts_dataloader, test_acts_dataloader = get_dataloaders_from_datasets_AE(train_acts_dataset, test_acts_dataset, validation_mode, validation_split,
                                  global_args['ae_pretrain_batch_size'], global_args['num_workers_ae_pretraining'], DataLoader)

    auto_encoder_trainer = CentralizedAETrainer()

    optimizer, scheduler = get_ae_pretrain_optimizer_and_scheduler(auto_encoder_model, global_args)
    loss_fn = torch.nn.MSELoss()

    ae_experiment_results = ExperimentResultsAE(validation_mode)
    model_save_dir = prepare_ae_dir(ae_signature)
    model_save_path = os.path.join(model_save_dir, 'ae_model')

    # Validation State Variables
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting AE Pre-training\n")

    start_time = time.time()

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    with trange(global_args['ae_pretrain_epochs'], unit='epoch') as epochs:
        for epoch_nr in epochs:

            train_loss = auto_encoder_trainer.train_epoch(
                experiment_results=ae_experiment_results,
                auto_encoder=auto_encoder_model,
                base_model=base_model,
                split_layer=global_args['split_layer'],
                device=device,
                dataloader=train_acts_dataloader,
                epoch_nr=epoch_nr,
                optimizer=optimizer,
                loss_fn=loss_fn
            )
            # print(f"Epoch {epoch_nr} --- Train --- Reconstruction Loss: {train_loss:.3f}")

            scheduler.step()

            if val_acts_dataloader is not None and validation_mode in ['early_saving', 'early_stopping']:
                with torch.no_grad():
                    val_loss = auto_encoder_trainer.test_epoch(
                        experiment_results=ae_experiment_results,
                        auto_encoder=auto_encoder_model,
                        base_model=base_model,
                        split_layer=global_args['split_layer'],
                        device=device,
                        dataloader=val_acts_dataloader,
                        epoch_nr=epoch_nr,
                        loss_fn=loss_fn
                    )
                    # print(f"Epoch {epoch_nr} --- Validation --- Reconstruction Loss: {train_loss:.3f}")

                    # Check improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # print(f"Validation loss improved to {best_val_loss:.6f}. Saving model to {model_save_path}...")
                        if model_save_path:
                            torch.save(auto_encoder_model.state_dict(), model_save_path)
                    else:
                        patience_counter += 1

                    if validation_mode == 'early_stopping' and patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch_nr} epochs.")
                        break

                    epochs.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                        'Val Loss': f'{val_loss:.4f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.1e}',
                    })

            # Option 1: No Val (or just logging test set per epoch like original code)
            elif validation_mode == 'none':
                with torch.no_grad():

                    test_loss = auto_encoder_trainer.test_epoch(
                        experiment_results=ae_experiment_results,
                        auto_encoder=auto_encoder_model,
                        base_model=base_model,
                        split_layer=global_args['split_layer'],
                        device=device,
                        dataloader=test_acts_dataloader,
                        epoch_nr=epoch_nr,
                        loss_fn=loss_fn
                    )
                    ae_experiment_results.final_test_metric = test_loss
                    # print(f"Epoch {epoch_nr} --- Test --- Loss: {test_loss:.3f}\n")

                    epochs.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                        'Test Loss': f'{test_loss:.4f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.1e}',
                    })

            save_ae_experiment_results(ae_experiment_results, filename_from_signature(ae_signature))

    print(f'Finished training AE.')


    if validation_mode in ['early_saving', 'early_stopping']:
        # print(f"Loading best model from {model_save_path} for final testing...")
        try:
            auto_encoder_model.load_state_dict(torch.load(model_save_path))
        except FileNotFoundError:
            print("Warning: Best model file not found. Using current weights.")

        print('Final AE Test Set Evaluation')
        with torch.no_grad():
            final_test_loss = auto_encoder_trainer.test_epoch(
                experiment_results=None,
                auto_encoder=auto_encoder_model,
                base_model=base_model,
                split_layer=global_args['split_layer'],
                device=device,
                dataloader=test_acts_dataloader,
                epoch_nr=global_args['ae_pretrain_epochs'],
                loss_fn=loss_fn
            )
            ae_experiment_results.final_test_metric = final_test_loss
            print(f"Final Test Metric from saved model --- Loss: {final_test_loss:.3f}")

    if global_args['ae_save_final_weights']:
        save_ae_with_signature(auto_encoder_model, ae_signature)

    save_ae_experiment_results(ae_experiment_results, filename_from_signature(ae_signature))

    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nAE pre-training completed in: {duration:.2f} seconds\n")

    ae_experiment_results_json = ae_experiment_results.to_json()
    ae_experiment_results_json['reused_AE'] = False

    return auto_encoder_model, ae_experiment_results_json

def extract_activations(base_model, split_layer, dataloader, device, raw_cifar=False, *, cache_path, seed=42):
    """Stream frozen features into a complete, atomically published mmap cache."""
    return write_activation_cache(base_model, split_layer, dataloader, device,
                                  cache_path, raw_cifar=raw_cifar, seed=seed)
