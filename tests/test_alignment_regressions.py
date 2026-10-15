"""Alignment loss and repeated decoder synchronization regressions."""
import copy

import pytest
import torch
from torch import nn

from ae_psl.ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from ae_psl.models.auto_encoder import BaseViTAE
from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer
from ae_psl.trainers.implementations.experiment_results import ExperimentResults


def test_ga_metrics_weight_samples_in_final_short_batch():
    ae = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ae.weight.zero_()
    loss = CentralizedAETrainer()._perform_epoch(None, ae, 'cpu',
        [(torch.ones(2, 1),), (torch.tensor([[3.]]),)], 0, None, nn.MSELoss())
    assert loss == pytest.approx(11 / 3)


def test_ga_rejects_empty_evaluation_instead_of_perfect_loss():
    with pytest.raises(ValueError, match='samples'):
        CentralizedAETrainer()._perform_epoch(None, nn.Identity(), 'cpu', [], 0, None, nn.MSELoss())


class AlignmentClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.ae_module = BaseViTAE()
        self.ae_module.patch_encoder = nn.Linear(4, 2)
        self.ae_module.patch_decoder = nn.Linear(2, 4)

    def forward(self, x):
        return self.backbone(x)


def test_iterative_csa_redistributes_shared_decoder_and_retains_local_encoders():
    torch.manual_seed(11)
    clients = {0:AlignmentClient(), 1:AlignmentClient()}
    server = copy.deepcopy(clients[0])
    before = [copy.deepcopy(client.backbone.state_dict()) for client in clients.values()]
    optimizers = {cid:torch.optim.SGD(client.ae_module.parameters(), lr=.1) for cid,client in clients.items()}
    results = ExperimentResults('none')
    trainer = SASLTrainer()
    trainer.client_specific_alignment('cpu', server,
        torch.optim.SGD(server.parameters(), lr=.1),
        [{0:(torch.randn(2, 3, 4), torch.tensor([0, 1])),
          1:(torch.randn(3, 3, 4)+2, torch.tensor([0, 1, 0]))}],
        clients, True, {}, {}, optimizers, {}, 1, results, -2,
        {'nr_of_clients':2, 'dataset':'fixture', 'iterative_client_specific_alignment':True})
    for cid, client in clients.items():
        for name, value in client.ae_module.patch_decoder.state_dict().items():
            torch.testing.assert_close(value, server.ae_module.patch_decoder.state_dict()[name])
        for name, value in client.backbone.state_dict().items():
            torch.testing.assert_close(value, before[cid][name])
    assert not torch.equal(clients[0].ae_module.patch_encoder.weight, clients[1].ae_module.patch_encoder.weight)
    communication = results.to_json()['incoming_client_communication_overhead_in_mb']
    assert communication["-2"] > 0


def test_identity_at_split_zero_trains_server_without_client_gradient():
    from types import SimpleNamespace
    from torchvision.models.vision_transformer import VisionTransformer
    from ae_psl.models.vision_transformer.implementations.unimodal.image_classification.models import get_split_model
    from ae_psl.orchestrator import setup_orchestrator_parser
    from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space
    vit = VisionTransformer(image_size=16, patch_size=8, num_layers=2, num_heads=2,
                            hidden_dim=8, mlp_dim=16, num_classes=3)
    for parameter in vit.parameters():
        parameter.requires_grad = False
    for parameter in vit.heads.parameters():
        parameter.requires_grad = True
    client, server, needs_gradient = get_split_model(
        SimpleNamespace(vit=vit, use_lora=True), BaseViTAE(), 0, 3, 'cpu', False)
    assert not needs_gradient
    before = copy.deepcopy(client.state_dict())
    args, _ = namespace_to_global_args_and_search_space(setup_orchestrator_parser().parse_args(['--dataset', 'food101']))
    args.update(nr_of_clients=1, dataset='fixture', freeze_clientside=False)
    result = SASLTrainer()._single_epoch('cpu', server, torch.optim.SGD(server.parameters(), lr=.1),
        [{0:(torch.randn(2,3,16,16),torch.tensor([0,1]))}], {0:client}, needs_gradient,
        {0:torch.optim.SGD(client.parameters(), lr=.1)}, {}, 1, 0, args)
    assert result[0] > 0
    assert server.heads.head.weight.abs().sum() > 0
    for name, value in client.state_dict().items():
        torch.testing.assert_close(value, before[name])
