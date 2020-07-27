from .AbstractEval import AbstractEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t
import numpy as np

class TorchEval(AbstractEval):
    def eval_Constant(self, node: DagNode):
        return t.Tensor([node.value])

    def eval_Add(self, a, b, node: DagNode):
        return a + b

    def eval_Sub(self, a, b, node: DagNode):
        return a - b

    def eval_Mul(self, a, b, node: DagNode):
        return a * b

    def eval_Concat(self, a, b, node: DagNode):
        if node.choice == 0:
            #return t.cat([a, b], dim= node.concat_dim)
            return (np.append([a],[b]))

        if node.choice == 1:
            #return np.append([a],[b])
            return np.row_stack((a,b))

    def eval_Round(self, a, prec, node: DagNode):
        scale = 2.0**prec
        return IntRound(a * scale)/scale

    def eval_Relu(self, a, node: DagNode):
        return t.relu(a)

    def eval_Len(self, a, node: DagNode):
        return len(a)

    def eval_Select(self, a, node: DagNode):
        return a[node.index]

    def eval_Reduce(self, a, node: DagNode):
        sum = t.sum(a, dim=node.reduce_dim)
        return sum
