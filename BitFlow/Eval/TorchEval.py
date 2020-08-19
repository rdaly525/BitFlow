from .AbstractEval import AbstractEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t
from BitFlow.node import Add, Constant
import numpy as np


from .AbstractEval import AbstractEval
from ..node import DagNode, Dag, LookupTable
from ..torch.functions import IntRound
import torch as t
import math

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

    def eval_BitShift(self, a, b, node: DagNode):
        return a * b

    def eval_Round(self, a, prec, rng, node: DagNode):
        scale = 2.0 ** (prec)
        precise = IntRound(an * scale) / scale
        precise = precise.float()

        # saturate value to range
        prec = prec.float()
        rng = rng.float()
        min_val = -1 * (2 ** (prec + rng - 1)) * \
                  2 ** -prec
        max_val = (2 ** (prec + rng - 1) - 1) * 2 ** -prec

        if t.numel(precise[precise > max_val]) > 0:
            self.saturation = self.saturation + (t.sum(precise[precise > max_val] -
                                                       max_val) / t.numel(precise[precise > max_val])) * 2 ** prec

        if t.numel(precise[precise < min_val]) > 0:
            self.saturation = self.saturation + (t.sum(t.abs(precise[precise < min_val] -
                                                             min_val)) / t.numel(
                precise[precise < min_val])) * 2 ** prec

        # if rng <= 0.01:
        #     print(f"{rng}.{prec}")
        #     print(f"=>  [{min_val, max_val}]")
        #     print(a)
        #     print(self.saturation)
        #     assert 0

        precise[precise > max_val] = max_val
        precise[precise < min_val] = min_val

        return precise

    def eval_Select(self, a, node: DagNode):
        if len(a.shape) == 1:
            return a[node.index]
        else:
            print(a)
            return a[:, node.index]

    # def eval_Select(self, a, node: DagNode):
    #     return a[node.index]

    def eval_Relu(self, a, node: DagNode):
        return t.relu(a)

    def eval_Tanh(self, a, node: DagNode):
        return t.tanh(a)

    def eval_Reduce(self, a, node: DagNode):
        sum = t.sum(a, dim=node.reduce_dim)
        # pass in the size we are reducing over
        return sum

    def eval_LookupTable(self, a, node: LookupTable):
        if hasattr(node, 'lut'):
            return node.lut[a]
        else:
            return getattr(t, node.func.__name__)(a)

    def eval_Len(self, a, node: DagNode):
        return len(a)

    def eval_Concat(self, *args, node: DagNode):
        return t.stack(args, dim=node.concat_dim)

