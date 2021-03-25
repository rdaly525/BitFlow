from .AbstractEval import AbstractEval
from ..AA import AInterval
from ..node import DagNode, LookupTable


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
        return a[node.index]

    def eval_LookupTable(self, a, node: LookupTable):
        if hasattr(node, 'lut'):
            return node.lut[a]
        else:
            return AInterval(node.func(a.base), a.noise)
