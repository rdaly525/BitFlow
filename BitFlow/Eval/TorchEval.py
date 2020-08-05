from .AbstractEval import AbstractEval
from ..node import DagNode, Dag
from ..torch.functions import IntRound
import torch as t


class TorchEval(AbstractEval):
    def __init__(self, dag: Dag):
        super().__init__(dag)
        self.saturation = t.Tensor([0.])

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

        if type(precise) is t.LongTensor:
            print(precise)
            assert 0

        # saturate value to range
        max_prec = sum([2 ** -(p+1) for p in range(int(prec.item()))])

        min_val = -1 * (2 ** (rng - 1)) - max_prec
        max_val = 2 ** (rng - 1) - 1 + max_prec

        if t.numel(precise[precise > max_val]) > 0:
            self.saturation = self.saturation + t.sum(precise[precise > max_val] -
                                                      max_val)/t.numel(precise[precise > max_val])

        if t.numel(precise[precise < min_val]) > 0:
            self.saturation = self.saturation + t.sum(t.abs(precise[precise < min_val] -
                                                            min_val))/t.numel(precise[precise < min_val])

        precise[precise > max_val] = max_val
        precise[precise < min_val] = min_val

        return precise

    def eval_Select(self, a, node: DagNode):
        if len(a.shape) == 1:
            return a[node.index]
        else:
            return a[:, node.index]
