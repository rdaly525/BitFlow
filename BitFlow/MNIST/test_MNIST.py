from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Reduce, Concat

from BitFlow.MNIST.MNIST_library import linear_layer
from BitFlow.MNIST.run_MNIST import MNIST_dag


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X, W, bias, row, col, size)

    fig = Dag(outputs=[y], inputs=[X, W, bias])
    return fig


def test_linearlayer():
    row = 100
    col = 10
    size = 784
    print("Before gen")
    dag = gen_linearlayer(row, col, size)
    print("Enter")

    MNIST_dag(dag)

    return


test_linearlayer()
