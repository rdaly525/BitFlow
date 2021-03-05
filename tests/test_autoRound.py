import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Reduce
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.AddRoundNodes import NodePrinter1
from BitFlow.MNIST.MNIST_library import reduce_m, linear_layer, matrix_multiply, dot_product


def update_dag(dag):
    W = Input(name="W")
    O = Input(name="O")

    print("DAG before Round Nodes added:")
    printer = NodePrinter1()
    printer.run(dag)

    addedRoundNode = AddRoundNodes(W, O)
    newDag = addedRoundNode.doit(dag)

    print("DAG after Round Nodes added:")
    printer.run(newDag)

    return newDag


def gen_fig3():
    # (a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag


def gen_reduce_x():
    a = Input(name="a")
    b = Reduce(a, 0, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X, W, bias, row, col, size)

    fig = Dag(outputs=[y], inputs=[X, W, bias])
    return fig


def gen_ex1():
    # (a * b) + (b * c)
    a = Input(name="a")
    b = Input(name=b"b")
    c = Input(name="c")
    d = Mul(a, b, name="d")
    e = Mul(b, c, name="e")
    z = Add(e, d, name="z")

    dag = Dag(outputs=[z], inputs=[a, b, c])
    return dag


def gen_reduce_NEW(size):
    # a is a 1D, 5 size vector, z = sum along a + c
    a = Input(name="a")
    c = Input(name="c")
    # b = Reduce(a, 0, 5, name="b")
    b = reduce_m(a, size)

    z = Add(b, c, name="z")

    fig = Dag(outputs=[z], inputs=[a, c])
    return fig


def test_Reduce_NEW():
    print("###############TEST REDUCE_NEW###############")

    dag = gen_reduce_NEW(5)

    W = torch.tensor([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.])
    O = torch.tensor([20.])
    a = torch.tensor([0.23, 0.45, 0.56, 0.454, 0.23])
    c = torch.tensor(0.96)

    print("EVAL Before Rounding")
    evaluator = TorchEval(dag)
    res = evaluator.eval(a=a, c=c)
    print("y value with full precision")
    print(res)

    newDag = update_dag(dag)

    W1 = torch.tensor([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.])
    O1 = torch.tensor([20])
    a1 = torch.tensor([0.23, 0.45, 0.56, 0.454, 0.23])
    c1 = torch.tensor(0.96)

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "c": c1, "W": W1, "O": O1}

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_fig3():
    print("###############TEST FIG3###############")

    fig3 = gen_fig3()

    W = torch.tensor([20., 20., 20., 20., 20.])
    O = torch.tensor([20.])
    a, b = 10.45454524545452, 10.2323223232

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig3)
    res = evaluator.eval(a=a, b=b)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig3)

    W1 = torch.tensor([20., 20., 20., 20., 20.])
    O1 = torch.tensor([20])
    a1, b1 = 10.45454524545452, 10.2323223232

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "b": b1, "W": W1, "O": O1}

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_ex1():
    print("###############TEST EX1###############")
    fig3 = gen_ex1()

    W = torch.tensor([20., 20., 20., 20., 20.])
    O = torch.tensor([20.])
    a, b, c = 10.45454524545452, 10.2323223232, 8.231343

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig3)
    res = evaluator.eval(a=a, b=b, c=c)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig3)

    W1 = torch.tensor([20., 20., 20., 20., 20.])
    O1 = torch.tensor([20])
    a1, b1, c1 = 10.45454524545452, 10.2323223232, 8.231343

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "b": b1, "c": c1, "W": W1, "O": O1}

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_fig3_integers():
    fig3 = gen_fig3()
    print("###############TEST FIG3 INTEGERS###############")

    a, b = 3, 5

    W = torch.tensor([20., 20., 20., 20., 20.])
    O = torch.tensor([20.])

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig3)
    res = evaluator.eval(a=a, b=b)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig3)

    a1, b1 = 3, 5

    W1 = torch.tensor([20., 20., 20., 20., 20.])
    O1 = torch.tensor([20.])

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "b": b1, "W": W1, "O": O1}

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_poly_approx():
    print("###############POLY APPROX###############")
    tmp = caseStudy
    fig_casestudy = tmp.poly_approx()

    W = torch.tensor([15., 15., 15., 15., 15., 15., 15.,
                      15., 15., 15., 15., 15., 15., 15.])

    O = torch.tensor([15.])
    a, c = torch.tensor([1., 3., -6., -10., -1.]), 3.3

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig_casestudy)
    res = evaluator.eval(a=a, c=c)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig_casestudy)

    # W1 = torch.tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 10., 11., 12., 13., 14.])

    W1 = torch.tensor([15., 15., 15., 15., 15., 15., 15.,
                       15., 15., 15., 15., 15., 15., 15.])

    O1 = torch.tensor([15.])
    a1, c1 = torch.tensor([1., 3., -6., -10., -1.]), 3.3

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "c": c1, "W": W1, "O": O1}
    print(inputs1)

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_RGB_to_YCbCr():
    print("###############RGB to YCbCr###############")
    tmp = caseStudy
    fig_casestudy = tmp.RGB_to_YCbCr()

    W = torch.tensor(
        [15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.])

    O = torch.tensor([15., 15., 15.])
    a = torch.tensor([22., 103., 200.])

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig_casestudy)
    res = evaluator.eval(a=a)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig_casestudy)

    W1 = torch.tensor(
        [15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.])

    O1 = torch.tensor([15., 15., 15.])
    a1 = torch.tensor([22., 103., 200.])

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "W": W1, "O": O1}
    print(inputs1)

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_Matrix_Multiplication():
    print("###############Matrix Multiplication###############")
    tmp = caseStudy
    fig_casestudy = tmp.Matrix_Multiplication()

    W = torch.tensor(
        [15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.])

    O = torch.tensor([15., 15., 15., 15.])
    a = torch.tensor([[1.1, 2.2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig_casestudy)
    res = evaluator.eval(a=a, b=b)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig_casestudy)

    W1 = torch.tensor(
        [1., 1., 1., 1., 1., 1., 1., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.,
         15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15.])

    O1 = torch.tensor([1., 1., 1., 1.])
    a1 = torch.tensor([[1.1, 2.2], [3, 4]])
    b1 = torch.tensor([[5, 6], [7, 8]])

    print("EVAL After Rounding")
    inputs1 = {"a": a1, "b": b1, "W": W1, "O": O1}
    print(inputs1)

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_Linear_Layer():
    print("###############Linear Layer###############")
    tmp = gen_linearlayer(100, 10, 784)
    fig_casestudy = tmp

    X = torch.rand((100, 784))
    W = torch.rand((784, 10))
    O = torch.rand((10))

    bias = torch.rand((10))

    print("EVAL Before Rounding")
    evaluator = TorchEval(fig_casestudy)
    res = evaluator.eval(X=X, W=W, bias=bias)
    print("y value with full precision")
    print(res)

    newDag = update_dag(fig_casestudy)

    X1 = torch.trunc(X)
    W1 = torch.trunc(W)
    O1 = torch.trunc(O)

    print(W1)

    bias = torch.rand((10))

    print("EVAL After Rounding")
    inputs1 = {"X": X1, "W": W1, "bias": bias, "O": O1}
    # print(inputs1)

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


def test_ReduceDag():
    print("###############Linear Layer###############")
    tmp = gen_reduce_x()

    a = torch.tensor([1.232, 2.1212, 3.0239201323, 4.2323, 5.23, 6.231, 7])

    print("EVAL Before Rounding")
    evaluator = TorchEval(tmp)
    res = evaluator.eval(a=a)
    print(torch.sum(a, 0))
    print("y value with full precision")
    print(res)

    newDag = update_dag(tmp)

    W = torch.tensor(
        [4.])

    O = torch.tensor([15., 15., 15.])

    print("EVAL After Rounding")
    inputs1 = {"a": a, "W": W, "O": O}
    # print(inputs1)

    evaluator1 = TorchEval(newDag)

    y_val = evaluator1.eval(**inputs1)

    print("y value with rounded precision")
    print(y_val)

    return


# test_fig3()
# test_ex1()
# test_fig3_integers()
# test_poly_approx()
# test_RGB_to_YCbCr()
# test_Matrix_Multiplication()
test_Linear_Layer()
# test_Reduce_NEW()
# test_ReduceDag()
