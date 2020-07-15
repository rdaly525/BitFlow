from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output
from DagVisitor import Visitor, Transformer
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow


import torch


def gen_fig3():
    #(a*b) + 4 - b

    X=[1,2]

    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag



def test_fig3():

    dag = gen_fig3()

    BitFlow(dag, 6.)
    return


test_fig3()
