from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval, AbstractEval, TorchEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t
import typing as tp


def reduce_m(a, size):
    start = Constant(0)
    for i in range(0, size):
        start = Add(start, a[i])

    return start


def dot_product(a: DagNode, b: DagNode, size):
    common = Mul(a, b)
    return Reduce(common, reduce_dim=0)
    # return reduce_m(common, size)
    # return Reduce(common, 0, 0, name=f"{a.name}_dotproduct_{b.name}")


def matrix_multiply(a: DagNode, b: DagNode, row, col, size):
    output_array = []

    for i in range(0, row):
        ai = a[i]
        output_row_inputs = []
        for j in range(0, col):
            bj = b[:, j]

            output_element = dot_product(ai, bj, size)
            output_row_inputs.append(output_element)

        output_row = Concat(*output_row_inputs, concat_dim=0)

        output_array.append(output_row)

    return Concat(*output_array, concat_dim=0)


def linear_layer(X: DagNode, weight: DagNode, bias: DagNode, batch_dim, output_dim, size):
    y = matrix_multiply(X, weight, batch_dim, output_dim, size)

    bias_array = [bias for _ in range(batch_dim)]
    with_bias = Concat(*bias_array, concat_dim=0)

    return y + with_bias
