

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Len, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval,AbstractEval, TorchEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t


def dot_product(a: DagNode, b:DagNode):
    common = Mul(a,b)
    return Reduce(common, 0, name=f"{a.name}_dotproduct_{b.name}")


def matrix_multiply(a: DagNode, b:DagNode, row, col):

    #initialize first row of output array
    first_row = dot_product(a[0], b[:, 0])
    for m in range(1,col):
        first_row = Concat(first_row, dot_product(a[0], b[:, m]), concat_dim=0, choice=0)

    output_array = first_row
    output_row = dot_product(a[0], b[:, 0])


    for i in range(1,row):
        for j in range(0,col):

            if j == 0:
                output_row = dot_product(a[i], b[:, j])

            else:
                add_output_row = dot_product(a[i], b[:, j])
                output_row = Concat(output_row, add_output_row, concat_dim=0, choice=0)

        output_array = Concat(output_array, output_row,concat_dim=1, choice=1)

    return output_array



def linear_layer():
    pass
