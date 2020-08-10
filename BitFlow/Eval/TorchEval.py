from .AbstractEval import AbstractEval
from ..node import DagNode, Dag, LookupTable
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
        precise = precise.float()

        # saturate value to range
        max_prec = sum([2 ** -(p+1) for p in range(int(prec.item()))]) * 1.

        if rng > 0.:
            min_val = -1 * (2 ** (rng - 1)) - max_prec
            max_val = 2 ** (rng - 1) - 1 + max_prec

            if t.numel(precise[precise > max_val]) > 0:
                self.saturation = self.saturation + t.sum(precise[precise > max_val] -
                                                          max_val)/t.numel(precise[precise > max_val])

            if t.numel(precise[precise < min_val]) > 0:
                self.saturation = self.saturation + t.sum(t.abs(precise[precise < min_val] -
                                                                min_val))/t.numel(precise[precise < min_val])

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
            return a[:, node.index]

    def eval_LookupTable(self, a, node: LookupTable):
        return node.lut[a]


# PRECISION: tensor([18, 12, 17, 17, 10, 17, 17, 20, 11, 20, 18, 17, 18, 17, 17, 17, 17, 17,
#                    17, 18, 18, 17, 20, 20])
# RANGE: tensor([1, 9, 8, 1, 9, 9, 9, 1, 9, 6, 1, 8,
#                1, 8, 8, 1, 8, 1, 8, 1, 8, 8, 1, 6])

# ##### TEST SET ######
# accuracy: 1.0

# ##### TEST ULP ERROR ######
# AVG ERR: 0.25266602635383606
# MAX ERR: 0.671875

# ##### UFB TEST SET ######
# accuracy: 1.0

# ##### TRAIN SET ######
# accuracy: 1.0

# ##### UFB TRAIN SET ######
# accuracy: 1.0

# ##### MODEL DETAILS #####
# ERROR: 0.0005878292622417212
# AREA: 3594
# TIME: 323.4299898147583 SECONDS ELAPSED
# [tensor([113.6288]), tensor([-46.0696]), tensor([98.6954])]
