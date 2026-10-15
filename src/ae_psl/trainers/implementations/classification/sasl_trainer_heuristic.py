"""Serialized split training shared by Rand-Top-K, C3-SL and ADC.

Compression is explicit: evaluation mode does not disable it. Sparse indices and
codec shape metadata cross the boundary alongside values. C3's random keys are
pre-shared, checkpointed codec state; model/codec initialization is excluded from
per-batch communication, as for the AE. Low-bit quantization, if requested,
changes precision only; wire payloads remain float32.
"""
import json

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ae_psl.transport import transmit
from ae_psl.trainers.implementations.classification.sasl_trainer import (
    SASLTrainer, get_gpu_vit_transforms,
)


class SASLTrainerHeuristic(SASLTrainer):
    @staticmethod
    def _original_sample_predictions(client, predictions):
        if hasattr(client, 'expand_predictions'):
            return client.expand_predictions(predictions)
        return predictions

    def _single_epoch(self, device, server_model, server_optimizer, global_train_dataloader,
                      client_models, client_model_requires_any_grad, client_optimizers_main,
                      client_optimizers_ae, max_nr_of_batches_in_epoch, epoch_nr, global_args):
        ids = list(client_models)
        incoming = {cid: 0 for cid in ids}
        outgoing = {cid: 0 for cid in ids}
        counts = {cid: 0 for cid in ids}
        accuracy = {cid: (0, 0) for cid in ids}
        mses = {cid: [] for cid in ids}
        total_loss, total_correct = 0., 0
        train_clients = client_model_requires_any_grad and not global_args.get('freeze_clientside', False)
        transform = get_gpu_vit_transforms(True) if global_args['dataset'] == 'cifar100' else None

        for batch_nr, batch in enumerate(tqdm(global_train_dataloader)):
            if global_args.get('small_test_run') and batch_nr > 3:
                break
            if not batch:
                continue
            server_optimizer.zero_grad()
            originals, received, reconstructions, slices = {}, {}, [], {}
            total_samples = sum(len(labels) for _, labels in batch.values())
            cursor = 0
            for cid, (images, labels) in batch.items():
                client_optimizers_main[cid].zero_grad()
                images = images.to(device)
                if transform is not None:
                    images = transform(images / 255.)
                with torch.set_grad_enabled(train_clients):
                    features = client_models[cid](images)
                    payload, context = client_models[cid].encode_for_transport(features)
                # Metadata contains only shapes/lengths, never activations or labels.
                metadata = json.dumps(context, separators=(',', ':')).encode()
                context = json.loads(metadata)
                outgoing[cid] += 4 + len(metadata)
                wire = []
                for part in payload:
                    restored, size = transmit(part, requires_grad=part.is_floating_point())
                    wire.append(restored)
                    outgoing[cid] += size
                reconstructed = client_models[cid].decode_from_transport(wire, context)
                if reconstructed.shape == features.shape:
                    mses[cid].append(F.mse_loss(reconstructed.detach(), features.detach()).item())
                originals[cid], received[cid] = payload, wire
                slices[cid] = (cursor, cursor + len(reconstructed))
                cursor += len(reconstructed)
                reconstructions.append(reconstructed)
                counts[cid] += len(labels)

            predictions = server_model.forward_uncompressed(torch.cat(reconstructions))
            gradients = []
            for cid, (_, labels) in batch.items():
                start, end = slices[cid]
                local_logits, size = transmit(predictions[start:end], requires_grad=True)
                incoming[cid] += size
                expanded = self._original_sample_predictions(client_models[cid], local_logits)
                labels = labels.to(device)
                local_loss = F.cross_entropy(expanded, labels)
                local_gradient = torch.autograd.grad(local_loss, local_logits)[0]
                gradient, size = transmit(local_gradient * len(labels) / total_samples)
                gradients.append(gradient)
                outgoing[cid] += size
                scalar, size = transmit(local_loss)
                outgoing[cid] += size
                total_loss += scalar.item() * len(labels)
                correct = (expanded.argmax(1) == labels).sum().item()
                total_correct += correct
                old_correct, old_count = accuracy[cid]
                accuracy[cid] = (old_correct + correct, old_count + len(labels))

            predictions.backward(torch.cat(gradients))
            server_optimizer.step()
            if train_clients:
                for cid in batch:
                    outputs, gradients = [], []
                    for original, wire in zip(originals[cid], received[cid]):
                        if original.requires_grad:
                            gradient, size = transmit(wire.grad)
                            incoming[cid] += size
                            outputs.append(original)
                            gradients.append(gradient)
                    if outputs:
                        torch.autograd.backward(outputs, gradients)
                        client_optimizers_main[cid].step()

        return (total_loss, total_correct, sum(outgoing.values()), sum(incoming.values()),
                incoming, outgoing, accuracy, counts, mses, {cid:list(v) for cid,v in mses.items()})

    def _evaluate(self, client, server, loader, device, use_gpu_transform, compressed):
        client.eval()
        server.eval()
        loss, correct, count = 0., 0, 0
        transform = get_gpu_vit_transforms(False) if use_gpu_transform else None
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if transform is not None:
                    images = transform(images / 255.)
                features = client(images)
                if compressed:
                    features = client.compress_decompress(features)[0]
                predictions = server.forward_uncompressed(features)
                if compressed:
                    predictions = self._original_sample_predictions(client, predictions)
                loss += F.cross_entropy(predictions, labels, reduction='sum').item()
                correct += (predictions.argmax(1) == labels).sum().item()
                count += len(labels)
        if not count:
            raise ValueError('Evaluation received no samples')
        return {'loss': loss / count, 'acc': correct / count, 'samples': count}

    def test_epoch_global_evaluation(self, client_model, server_model, dataloader, device,
                                    experiment_results, epoch_nr, use_gpu_transform,
                                    compression_during_global_eval=False):
        metrics = self._evaluate(client_model, server_model, dataloader, device,
                                 use_gpu_transform, compression_during_global_eval)
        if experiment_results is not None:
            experiment_results.add_results(epoch_nr, metrics['acc'], True)
        return metrics['loss'], metrics['acc']

    def test_epoch_distributed(self, client_models, server_model, dataloaders, device,
                               experiment_results, epoch_nr, use_gpu_transform,
                               compression_during_local_eval=True):
        if not client_models:
            raise ValueError('Local evaluation requires at least one client')
        results = {cid: self._evaluate(client, server_model, dataloaders[cid], device,
                                      use_gpu_transform, compression_during_local_eval)
                   for cid, client in client_models.items()}
        loss = sum(result['loss'] for result in results.values()) / len(results)
        acc = sum(result['acc'] for result in results.values()) / len(results)
        if experiment_results is not None:
            experiment_results.add_results(epoch_nr, acc, True)
            experiment_results.distributed_eval[epoch_nr] = results
        return loss, acc, results
