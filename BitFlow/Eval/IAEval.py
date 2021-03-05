from .AbstractEval import AbstractEval
from ..node import DagNode

class IAEval(AbstractEval):
    def eval_Constant(self, node: DagNode):
        return node.value

    def eval_Add(self, a, b, node: DagNode):
        return a + b

    def eval_Sub(self, a, b, node: DagNode):
        return a - b

    def eval_Mul(self, a, b, node: DagNode):
        return a * b

    def eval_Select(self, a, node: DagNode):

        if (isinstance(node.index, tuple)):
            # return a[0,a.shape[0]][node.index[1]]
            return a[:, node.index[1]]

        return a[node.index]
