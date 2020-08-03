

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Len, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval,AbstractEval, TorchEval
from ..node import DagNode
from ..torch.functions import IntRound
import torch as t
import typing as tp


def dot_product(a: DagNode, b:DagNode):
    common = Mul(a,b)
    return Reduce(common, 0, name=f"{a.name}_dotproduct_{b.name}")

# def matrix_multiply(a: DagNode, b:DagNode, row, col):
#
#     #initialize first row of output array
#     first_row = dot_product(a[0], b[:, 0])
#     for m in range(1,col):
#         #first_row = Concat(first_row, dot_product(a[0], b[:, m]), concat_dim=0, choice=0)
#         first_row = Concat(first_row, dot_product(a[0], b[:, m]), concat_dim=0, choice=0)
#
#     output_array = first_row
#     output_row = dot_product(a[0], b[:, 0])
#
#     for i in range(1,row):
#         for j in range(0,col):
#
#             if j == 0:
#                 output_row = dot_product(a[i], b[:, j])
#
#             else:
#                 add_output_row = dot_product(a[i], b[:, j])
#                 output_row = Concat(output_row, add_output_row, concat_dim=0, choice=0)
#
#         output_array = Concat(output_array, output_row,concat_dim=1, choice=1)
#
#     return output_array

def matrix_multiply(a: DagNode, b:DagNode, row, col):

    # #initialize first row of output array
    # first_row = dot_product(a[0], b[:, 0])
    # for m in range(1,col):
    #
    #     #first_row = Concat(first_row, dot_product(a[0], b[:, m]), concat_dim=0, choice=0)
    #     first_row = Concat(first_row, dot_product(a[0], b[:, m]), concat_dim=0, choice=0)

    output_array = []


    for i in range(0,row):
        ai= a[i]
        output_row_inputs = []
        for j in range(0,col):


            bj = b[:,j]
            output_element = dot_product(ai, bj)
            output_row_inputs.append(output_element)

        output_row = Concat(*output_row_inputs, concat_dim=0)

        output_array.append(output_row)

    return Concat(*output_array,concat_dim=0)


#CLASS torch.nn.Linear(in_features, out_features, bias=True)
#Applies a linear transformation to the incoming data: y = x*W^T + b

# in_features – size of each input sample (i.e. size of x)
# out_features – size of each output sample (i.e. size of y)
# bias – If set to False, the layer will not learn an additive bias. Default: True


# def linear_layer(a: DagNode, b: DagNode):
#     y = x.matmul(m.weight.t()) + m.bias  # y = x*W^T + b

#
def linear_layer(X: DagNode, W: DagNode, bias:DagNode, batch_dim, output_dim):
    y = matrix_multiply(X, W, batch_dim, output_dim)

    bias_array = [bias for _ in range(batch_dim)]
    with_bias = Concat(*bias_array, concat_dim=0)

    return y + with_bias