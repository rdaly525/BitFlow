from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Concat, Softmax

from BitFlow.MNIST.MNIST_library import linear_layer
from BitFlow.MNIST.run_MNIST import MNIST_dag
from BitFlow.MNIST.Bitflow_with_MNIST import BitFlow
import torch




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


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    weight = Input(name="weight")
    bias = Input(name="bias")
    _concat_add__concat_bias = linear_layer(X, weight, bias, row, col, size)

    #z = Softmax(y,name="z")

    fig = Dag(outputs=[_concat_add__concat_bias], inputs=[X, weight, bias])
    return fig


def test_linearlayer():
    row = 100
    col = 10
    size = 784
    dag = gen_linearlayer(row, col, size)


    params = dict(
        training_size=60000,
        testing_size=2000,
        epochs=5,
        batch_size=100,
        lr=1e-4
    )


    bf = BitFlow(dag, {"_concat_add__concat_bias":10.}, {
        'X': (0., 10.), 'weight': (0., 10.), 'bias': (0., 10.)}, **params)

    # bf = BitFlow(dag, {"y0": 10., "y1": 10., "y2": 10., "y3": 10., "y4": 10., "y5": 10., "y6": 10., "y7": 10., "y8": 10., "y9": 10.}, {
    #     'X': (0., 10.), 'weight': (0., 10.), 'bias': (0., 10.)}, **params)

    # Sample Matrix Product
    test = {"X": 10., "weight": 10., "bias": 10., "W": bf.W, "O": bf.O}
    print(bf.model(**test))

    return

    #     bf = BitFlow(dag, {"y": torch.ones(10)},{
    #         'X': torch.ones(100,784), 'weight': torch.ones(784,10), 'bias': torch.ones(10)}, **params)
    #

    return



test_linearlayer()
#test_rgb_case_study()