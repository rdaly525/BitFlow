from BitFlow.torch.functions import Round
import torch as t

def test_round():
    x = t.Tensor([1.2, 3.7])
    assert t.all(Round(x) == t.Tensor([1.0, 4.0]))
