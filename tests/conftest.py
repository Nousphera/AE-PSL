import gc
import pytest
import torch


@pytest.fixture(autouse=True)
def bounded_cpu_execution():
    torch.set_num_threads(1)
    yield
    gc.collect()
