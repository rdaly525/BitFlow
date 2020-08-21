from .AbstractEval import AbstractEval
from ..node import DagNode, LookupTable
from ..torch.functions import IntRound
import torch as t
import numpy as np

class NumEval(AbstractEval):
    def eval_Constant(self, node: DagNode):
        return node.value

    def eval_Len(self, node: DagNode):
        return node.size

    def eval_Add(self, a, b, node: DagNode):
        return a + b

    def eval_Sub(self, a, b, node: DagNode):
        return a - b

    def eval_Mul(self, a, b, node: DagNode):

        return a * b

    def eval_Select(self, a, node: DagNode):
        if(isinstance(node.index,tuple)):

            #return a[0,a.shape[0]][node.index[1]]
            return a[:,node.index[1]]
        #print(node.index)
        return a[node.index]

    def eval_Relu(self, a, node: DagNode):
        return t.relu(a)

    def eval_Tanh(self, a, node: DagNode):
        return t.tanh(a)

    def eval_Reduce(self, a, node: DagNode):
        sum = t.sum(a, dim=node.reduce_dim)
        # pass in the size we are reducing over
        return sum


    def eval_Concat(self, *args, node: DagNode):
        return t.stack(args, dim=node.concat_dim)


    def eval_LookupTable(self, a, node: LookupTable):
        if hasattr(node, 'lut'):
            return node.lut[a]
        else:
            return node.func(a)