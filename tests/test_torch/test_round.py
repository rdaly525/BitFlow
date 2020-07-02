from BitFlow.torch.functions import IntRound
import torch as t

def test_round():
    x = t.Tensor([1.2, 3.7])
    assert t.all(IntRound(x) == t.Tensor([1.0, 4.0]))
