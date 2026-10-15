import torch
import pytest
from ae_psl.models.lora_layer import LoRALinear
from ae_psl.utils.scheduler_utils import get_scheduler


def test_biasless_lora_preserves_base_at_initialization():
    base = torch.nn.Linear(3, 4, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(2, 3)
    torch.testing.assert_close(layer(x), base(x))


def test_constant_scheduler_is_supported():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = get_scheduler(optimizer, {"scheduler": "constant"})
    assert scheduler.get_last_lr() == [0.001]

from types import SimpleNamespace
from ae_psl.models.auto_encoder import initialize_AE
from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer


def new_ae():
    return initialize_AE({'ae_type': '2_layer_MLP_(d_d_d_z)', 'ae_latent_dim': 64,
                          'ae_initialization': 'xavier'}, 768)


def test_csa_updates_with_default_frozen_dft_and_averages_only_decoders():
    torch.manual_seed(42)
    clients = {i: SimpleNamespace(ae_module=new_ae(), forward=lambda x: x) for i in range(2)}
    before = [{k: v.clone() for k, v in c.ae_module.state_dict().items()} for c in clients.values()]
    optimizers = {i: torch.optim.Adam(c.ae_module.parameters(), lr=.01) for i, c in clients.items()}
    dummy = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
    batch = {0: (torch.randn(1, 50, 768), torch.zeros(1)),
             1: (torch.randn(2, 50, 768) + 2, torch.zeros(2))}
    _, decoder = SASLTrainer()._single_aggregated_warmup_epoch(
        'cpu', None, dummy, [batch], clients, True, {}, optimizers, 1, 0,
        {'nr_of_clients': 2, 'dataset': 'synthetic', 'concurrent_mse_alignment': False})
    for i, client in clients.items():
        assert any(not torch.equal(v, before[i][k]) for k, v in client.ae_module.state_dict().items())
    assert set(decoder.state_dict()) == set(clients[0].ae_module.patch_decoder.state_dict())
    for k, v in decoder.state_dict().items():
        torch.testing.assert_close(v, sum(c.ae_module.patch_decoder.state_dict()[k] for c in clients.values()) / 2)


def test_global_metrics_weight_final_short_batch():
    class Client:
        def eval(self): pass
        def forward(self, x): return x
    class Server:
        def eval(self): pass
        def forward_uncompressed(self, x): return x
    batches = [(torch.tensor([[5., 0.], [5., 0.]]), torch.tensor([0, 0])),
               (torch.tensor([[5., 0.]]), torch.tensor([1]))]
    loss, acc = SASLTrainer().test_epoch_global_evaluation(Client(), Server(), batches, 'cpu', None, 1, False, False)
    assert acc == pytest.approx(2/3)
    expected = torch.nn.functional.cross_entropy(torch.cat([x for x,y in batches]), torch.cat([y for x,y in batches]))
    assert loss == pytest.approx(expected.item())

from ae_psl.utils.ae_registry_utils import filename_from_signature, verify_signature


def test_ae_cache_signature_covers_seed_and_training_settings():
    signature = dict(type='mlp',dataset='imagenet',dataset_proportion=1,model='vit_b_32',
                     split_layer=5,input_dim=768,latent_dim=24,seed=42)
    assert filename_from_signature(signature) != filename_from_signature({**signature,'seed':43})
    assert not verify_signature({'type':'mlp'},signature)


def test_local_evaluation_averages_clients_as_in_paper():
    class Client:
        def eval(self): pass
        def forward(self,x): return x
    class Server:
        def eval(self): pass
        def forward_uncompressed(self,x): return x
    batches = {0:[(torch.tensor([[5.,0.],[5.,0.]]),torch.tensor([0,0]))],
               1:[(torch.tensor([[5.,0.]]),torch.tensor([1]))]}
    _,accuracy,details = SASLTrainer().test_epoch_distributed(
        {0:Client(),1:Client()},Server(),batches,'cpu',None,1,False,False)
    assert accuracy == pytest.approx(.5)
    assert [details[i]['samples'] for i in range(2)] == [2,1]

from ae_psl.utils.scheduler_utils import get_optimzer_and_scheduler_for_seperate_ae_finetuning


def test_identity_autoencoder_needs_no_ae_optimizer():
    args = {'compression_method':'ae', 'concurrent_mse_alignment':False,
            'start_lr':.001, 'ae_finetune_lr':.001, 'scheduler':'constant',
            'ae_type':'2_layer_MLP_(d_d_d_z)', 'ae_latent_dim':768}
    class Client(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('weight', torch.nn.Parameter(torch.ones(1)))
            self.ae_module = initialize_AE(args,768)
    _,_,optimizer,_ = get_optimzer_and_scheduler_for_seperate_ae_finetuning(Client(),args)
    assert optimizer is None
