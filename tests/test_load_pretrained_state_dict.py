import pytest
import torch

from models.metric import model_network_metric as model_network
from models.metric import model_train_metric as md


def create_model():
    model = md.Model('tmp_model', 'data/tmp/path', dim=3, source=[0,0,0], device='cpu')
    model.B = torch.randn(128, 3)
    model.network = model_network.NN(model.Params['Device'], model.dim, model.B)
    return model


def test_load_pretrained_state_dict():
    m1 = create_model()
    state = {
        'model_state_dict': m1.network.state_dict(),
        'B_state_dict': m1.B.clone(),
    }
    m2 = md.Model('tmp_model', 'data/tmp/path', dim=3, source=[0,0,0], device='cpu')
    m2.load_pretrained_state_dict(state)

    for p1, p2 in zip(m1.network.parameters(), m2.network.parameters()):
        assert torch.allclose(p1, p2)
    assert torch.equal(m2.B, m1.B)
    assert m2.B.device.type == 'cpu'
