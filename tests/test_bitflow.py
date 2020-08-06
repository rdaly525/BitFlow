from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch


def gen_fig3():
    # (a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4.3, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag


def gen_ex1():
    # (a * b) + (b * c)
    a = Input(name="a")
    b = Input(name="b")
    c = Input(name="c")
    d = Mul(a, b, name="d")
    e = Mul(b, c, name="e")
    z_1 = Add(e, d, name="z_1")
    z_2 = Add(a, d, name="z_2")

    dag = Dag(outputs=[z_1, z_2], inputs=[a, b, c])
    return dag

def gen_MNIST():
    # (a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4.3, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag

def test_fig3():

    dag = gen_fig3()

    BitFlow(dag, {"z": 8.}, {'a': (-3., 2.), 'b': (4., 8.)})
    return


def test_ex1():
    dag = gen_ex1()

    BitFlow(dag, {"z_1": 5., "z_2": 8.}, {
            'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)})
    return


def RGB_to_YCbCr():
    r = Input(name="r")
    g = Input(name="g")
    b = Input(name="b")

    col_1 = Add(Add(Mul(Constant(.299, name="C1"), r), Mul(Constant(.587, name="C2"), g)),
                Mul(Constant(.114, name="C3"), b), name="col_1")
    col_2 = Add(Add(Mul(Constant(-.16875, name="C4"), r), Mul(Constant(-.33126, name="C5"), g)),
                Mul(Constant(.5, name="C6"), b), name="col_2")
    col_3 = Add(Add(Mul(Constant(.5, name="C7"), r), Mul(Constant(-.41869, name="C8"), g)),
                Mul(Constant(-.08131, name="C9"), b), name="col_3")

    casestudy_dag = Dag(outputs=[col_1, col_2, col_3], inputs=[r, g, b])

    return casestudy_dag


def test_rgb_case_study():
    dag = RGB_to_YCbCr()

    params = dict(
        training_size=2000,
        testing_size=200,
        epochs=50,
        batch_size=16,
        lr=1e-4
    )

    bf = BitFlow(dag, {"col_1": 10., "col_2": 10., "col_3": 10.}, {
        'r': (-10., 10.), 'b': (-10., 10.), 'g': (-10., 10.)}, **params)

    # Sample Matrix Product
    test = {"r": 2., "g": 4., "b": -3., "W": bf.W, "O": bf.O}
    print(bf.model(**test))

    return

test_fig3()