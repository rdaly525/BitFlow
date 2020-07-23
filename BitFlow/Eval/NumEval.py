from .AbstractEval import AbstractEval
from ..node import DagNode

class NumEval(AbstractEval):
    def eval_Constant(self, node: DagNode):
        return node.value

    def eval_Add(self, a, b, node: DagNode):
        return a + b

    def eval_Sub(self, a, b, node: DagNode):
        return a - b

    def eval_Mul(self, a, b, node: DagNode):
        return a * b

    def eval_Select(self, a, node: DagNode):
        return a[node.index]

    def eval_Relu(self, a, node: DagNode):
        if a > 0:
            return a
        else:
            return 0

    def eval_Reduce(self, a, node: DagNode):

        sum = a[0]

        i = 1
        for i in range(1, len(a)):
            sum = sum + a[i]

        return sum
