from .AbstractEval import AbstractEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t


class TorchEval(AbstractEval):
    def eval_Constant(self, node: DagNode):
        return t.Tensor([node.value])

    def eval_Add(self, a, b, node: DagNode):
        return a + b

    def eval_Sub(self, a, b, node: DagNode):
        return a - b

    def eval_Mul(self, a, b, node: DagNode):
        return a * b

    def eval_Round(self, a, prec, rng, node: DagNode):
        scale = 2.0**prec
        precise = IntRound(a * scale)/scale

        if rng is None:
            return precise

        # saturate value to range
        max_prec = sum([2 ** -(p+1) for p in range(int(prec.item()))])

        min_val = -1 * (2 ** (rng - 1)) - max_prec
        max_val = 2 ** (rng - 1) - 1 + max_prec

        precise[precise > max_val] = max_val
        precise[precise < min_val] = min_val
        return precise

    def eval_Select(self, a, node: DagNode):
        if len(a.shape) == 1:
            return a[node.index]
        else:
            return a[:, node.index]
