from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Len, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer
from BitFlow.MNIST.run_MNIST import MNIST_dag



def gen_linearlayer(row,col):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X,W,bias,row,col)

    fig = Dag(outputs=[y], inputs=[X,W,bias])
    return fig


def test_linearlayer():
    row = 100
    col = 10
    dag = gen_linearlayer(row,col)


    MNIST_dag(dag)
    #MNIST_dag(dag)
    return


# def test_ex1():
#     dag = gen_ex1()
#
#     BitFlow(dag, {"z_1": 5., "z_2": 8.}, {
#             'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)})
#     return


test_linearlayer()