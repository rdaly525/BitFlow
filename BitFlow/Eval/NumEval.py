from .AbstractEval import AbstractEval
from ..node import DagNode
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
        return a[node.index]

    def eval_Relu(self, a, node: DagNode):
        return t.relu(a)

    def eval_Tanh(self, a, node: DagNode):
        return t.tanh(a)

    def eval_Reduce(self, a, node: DagNode):
        sum = t.sum(a, dim=node.reduce_dim)
        # pass in the size we are reducing over
        return sum

        # def eval_Reduce(self, a, node: DagNode):
        #         start = Constant(0,name = "start_")
        #         for i in range(0,node.size):
        #             start = Add(start,node[i], name = "start_"+str(i))
        #             print(start)
        #         node = start

    def eval_Len(self, a, node: DagNode):
        return len(a)

    def eval_Concat(self, *args, node: DagNode):
        return t.stack(args, dim=node.concat_dim)

    def eval_Round(self, a, prec, node: DagNode):
        scale = 2.0 ** prec
        return IntRound(a * scale) / scale