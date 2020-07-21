from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch


def gen_fig3():
    #(a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4.3, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag


def gen_ex1():
    #(a * b) + (b * c)
    a = Input(name="a")
    b = Input(name="b")
    c = Input(name="c")
    d = Mul(a, b, name="d")
    e = Mul(b, c, name="e")
    z_1 = Add(e, d, name="z_1")
    z_2 = Add(a, d, name="z_2")

    dag = Dag(outputs=[z_1, z_2], inputs=[a, b, c])
    return dag


def test_fig3():

    dag = gen_fig3()

    BitFlow(dag, {"z": 8.}, {'a': (-3., 2.), 'b': (4., 8.)})
    return


def test_ex1():
    dag = gen_ex1()

    BitFlow(dag, {"z_1": 5., "z_2": 8.}, {
            'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)})
    return
