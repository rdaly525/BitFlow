import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
#from BitFlow.AddRoundNodes import NodePrinter
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer
import sys


def gen_display():
    a = Input(name="a")
    z = a
    fig = Dag(outputs=[z], inputs=[a])
    return fig


def gen_multiply():
    a = Input(name="a")
    b = Input(name="b")
    z = Mul(a, b, name="z")

    fig = Dag(outputs=[z], inputs=[a, b])
    return fig


def gen_add():
    a = Input(name="a")
    b = Input(name="b")
    # c = Input(name="c")
    z = Add(a, b, name="z")

    fig = Dag(outputs=[z], inputs=[a, b])
    return fig


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X, W, bias, row, col, size)

    fig = Dag(outputs=[y], inputs=[X, W, bias])
    return fig


#
# def gen_length():
#     a = Input(name="a")
#     z = Len(a)
#
#     fig = Dag(outputs=[z], inputs=[a])
#     return fig

def gen_dotproduct(size):
    a = Input(name="a")
    b = Input(name="b")
    z = dot_product(a, b, size)

    fig = Dag(outputs=[z], inputs=[a, b])
    return fig


def gen_matrix_multiply():
    a = Input(name="a")
    b = Input(name="b")

    z = matrix_multiply(a, b)

    fig = Dag(outputs=[z], inputs=[a, b])
    return fig


def gen_matrix_mul(row, col, size):
    a = Input(name="a")
    b = Input(name="b")

    z = matrix_multiply(a, b, row, col, size)

    fig = Dag(outputs=[z], inputs=[a, b])
    return fig


def gen_relu():
    a = Input(name="a")
    b = Relu(a, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def gen_reduce_NEW():
    a = Input(name="a")
    b = Reduce(a, 0, 5, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def gen_reduce_x():
    a = Input(name="a")
    b = Reduce(a, 0, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def gen_reduce_y():
    a = Input(name="a")
    b = Reduce(a, 1, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def gen_select():
    a = Input(name="a")
    b = a[:,2]

    fig = Dag(outputs=[b], inputs=[a])
    return fig

def gen_concat():
    a = Input(name="a")

    hello = []
    for i in range(3):
        hello.append(a[i,2])

    b = Concat(*hello,concat_dim=0)


    fig = Dag(outputs=[b], inputs=[a])
    return fig

def gen_reduce_z():
    a = Input(name="a")
    b = Reduce(a, 2, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def test_linearlayer():
    row = 1
    col = 10
    tmp = gen_linearlayer(row, col, 784)
    evaluator = NumEval(tmp)
    X = torch.rand((1, 784))
    W = torch.rand((784, 10))

    bias = torch.rand((10))

    bias_array = [bias for _ in range(1)]
    with_bias = torch.stack(bias_array,dim=0)


    res = evaluator.eval(X=X, W=W, bias=bias)

    # dag_grapher = DAGGrapher(list(evaluator.dag.roots()))
    # dag_grapher.run(evaluator.dag)
    # dag_grapher.draw()
    #
    # print("here")


    gold = X @ W + with_bias
    print(gold)
    print(res)

    #assert torch.all(torch.eq(res,gold))

    # print(res.shape)
    # print(gold.shape)
    # for i in range(len(res)):
    #     for j in range(len(res[i])):
    #         print(res[i][j],gold[i][j])
    #         assert torch.eq(res[i][j],gold[i][j])

    return


# def test_length():
#     tmp = gen_length()
#     evaluator = NumEval(tmp)
#     a = torch.tensor([1])
#     res = evaluator.eval(a=a)
#     gold = 1
#     assert res==gold
#
#     a = torch.tensor([1,2])
#     res = evaluator.eval(a=a)
#     gold = 2
#     assert res == gold
#
#     a = torch.tensor([1,3,4])
#     res = evaluator.eval(a=a)
#     gold = 3
#     assert res == gold
#
#     a = torch.tensor([[1,3,4],[1,2,3]])
#     res = evaluator.eval(a=a)
#     gold = 2
#     assert res == gold
#
#     a = torch.tensor([[1,3],[1,2],[1,2]])
#     res = evaluator.eval(a=a)
#     gold = 3
#     assert res == gold
#
#     a = torch.tensor([[1,3],[1,2],[1,2]])
#     a = a[0]
#     res = evaluator.eval(a=a)
#     gold = 2
#     assert res == gold
#
#     a = torch.tensor([[[2,2]]])
#     a = a[0][0]
#     res = evaluator.eval(a=a)
#     gold = 2
#     assert res == gold
#
#     return

def test_dotproduct():
    tmp = gen_dotproduct(2)
    evaluator = NumEval(tmp)
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    res = evaluator.eval(a=a, b=b, concat_dim=0)
    gold = torch.tensor([11])
    # print(res)
    assert torch.all(torch.eq(res, gold))

    tmp = gen_dotproduct(5)
    evaluator = NumEval(tmp)
    a = torch.tensor([1, 2, 3, 0, -1])
    b = torch.tensor([3, 4, -1, 2.2, -1])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([9])
    assert torch.all(torch.eq(res, gold))

    tmp = gen_dotproduct(1)
    evaluator = NumEval(tmp)
    a = torch.tensor([1])
    b = torch.tensor([3])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([3])
    assert torch.all(torch.eq(res, gold))

    return


def test_matrix_mult():
    tmp = gen_matrix_mul(2, 2, 2)
    evaluator = NumEval(tmp)

    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = a @ b
    print(res)
    print(gold)
    assert torch.all(torch.eq(res, gold))
    print()

    tmp = gen_matrix_mul(2, 2, 2)
    evaluator = NumEval(tmp)

    a = torch.tensor([[1, 0], [3, 4]])
    b = torch.tensor([[5, 6], [0, 8]])

    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = a @ b
    assert torch.all(torch.eq(res, gold))
    print()

    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    tmp = gen_matrix_mul(3, 4, 2)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = a @ b
    assert torch.all(torch.eq(res, gold))
    print()

    a = torch.tensor([[1, 2, 6, 7], [3, 4, 5, 1], [5, 6, 1, 2]])
    b = torch.tensor([[1, 2, 6, 7], [3, 4, 5, 1], [5, 6, 1, 2], [3, 4, 5, 1]])
    tmp = gen_matrix_mul(3, 4, 4)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = a @ b
    assert torch.all(torch.eq(res, gold))
    print()

    a = torch.tensor([[1, 2, 6, 7], [3, 4, 5, 1], [5, 6, 1, 2], [3, 4, 5, 1]])
    b = torch.tensor([[1, 2, 4, 7]]).T
    tmp = gen_matrix_mul(4, 1, 4)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = a @ b
    assert torch.all(torch.eq(res, gold))

    return

def test_concat():
    tmp = gen_concat()
    evaluator = NumEval(tmp)
    #a = torch.tensor([[1,2],[4,5],[5,6]])
    a = torch.tensor([[100, 20000, 20, 400, 22, 11], [100, 20000, 20, 400, 22, 11], [100, 20000, 20, 400, 22, 11]])
    print(a)
    res = evaluator.eval(a=a)
    print(res)

    print("v")

    return

def test_select():
    tmp = gen_select()
    evaluator = NumEval(tmp)
    a = torch.tensor([[100, 20000, 20, 400, 22, 11],[100, 20000, 20, 400, 22, 11],[100, 20000, 20, 400, 22, 11]])
    res = evaluator.eval(a=a)

    print(res)

    return


def test_Relu():
    tmp = gen_relu()
    evaluator = NumEval(tmp)

    a = torch.tensor([100])
    res = evaluator.eval(a=a)
    gold = 100
    assert res == gold

    a = torch.tensor([-1000])
    res = evaluator.eval(a=a)
    gold = 0
    assert res == gold

    a = torch.tensor([100.2])
    res = evaluator.eval(a=a)
    gold = 100.2
    assert res == gold

    a = torch.tensor([3.14])
    res = evaluator.eval(a=a)
    gold = 3.14
    assert res == gold

    a = torch.tensor([3.14, 2.5, 3.2])
    res = evaluator.eval(a=a)
    gold = torch.tensor([3.14, 2.5, 3.2])
    assert torch.all(torch.eq(res, gold))

    a = torch.tensor([-3.14, -2.5, -3.2])
    res = evaluator.eval(a=a)
    gold = torch.tensor([0, 0, 0])
    assert torch.all(torch.eq(res, gold))

    a = torch.tensor([-3.14, -2.5, 5.53, -3.2])
    res = evaluator.eval(a=a)
    gold = torch.tensor([0, 0, 5.53, 0])
    assert torch.all(torch.eq(res, gold))

    return


def test_Reduce_NEW():
    tmp = gen_reduce_NEW()
    evaluator = NumEval(tmp)

    m = NodePrinter(tmp)

    a = torch.tensor([2, 4, 1, 3, 5])
    res = evaluator.eval(a=a)
    print(res)

    gold = 15
    assert res == gold
    return


def test_Reduce():
    tmp_x = gen_reduce_x()
    tmp_y = gen_reduce_y()
    tmp_z = gen_reduce_z()

    evaluator = NumEval(tmp_x)

    a = torch.tensor([2, 4, 1, 3])
    res = evaluator.eval(a=a)

    gold = 10
    assert res == gold

    a = torch.tensor([1, 2, 3, 9])
    res = evaluator.eval(a=a)
    gold = 15
    assert res == gold

    a = torch.tensor([1, 2, 3, 9, -1, 3.3, 4.6, 12.3])
    res = evaluator.eval(a=a)
    gold = 34.2
    assert res == gold

    a = torch.tensor([112.3])
    res = evaluator.eval(a=a)
    gold = 112.3
    assert res == gold

    a = torch.tensor([[112, 11], [1, 10]])
    res = evaluator.eval(a=a)
    gold = torch.tensor([113, 21])
    assert torch.all(torch.eq(res, gold))

    evaluator_y = NumEval(tmp_y)

    a = torch.tensor([[112, 11], [1, 10]])
    res = evaluator_y.eval(a=a)
    gold = torch.tensor([123, 11])
    assert torch.all(torch.eq(res, gold))

    evaluator_z = NumEval(tmp_z)

    a = torch.tensor([[[11, 112], [10, 11]], [[1, 2], [3, 10]]])
    res = evaluator_z.eval(a=a)
    gold = torch.tensor([[123, 21], [3, 13]])
    assert torch.all(torch.eq(res, gold))

    return


def test_add():
    tmp = gen_add()
    evaluator = NumEval(tmp)

    a = torch.tensor([100])
    b = torch.tensor([100])
    # c = torch.tensor([100])
    res = evaluator.eval(a=a, b=b)
    # print(res)
    gold = torch.tensor([200])
    assert res == gold

    a = torch.tensor([100, 200, 300])
    b = torch.tensor([100, 300, 400])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([200, 500, 700])
    assert torch.all(torch.eq(res, gold))
    return


#test_concat()
# test_Relu()
#test_Reduce()
# test_Reduce_NEW()
# test_dotproduct()
#test_select()
# test_length()
#test_matrix_mult()
# test_linearlayer()
# test_add() 


#test_select()
# test_matrix_mult()
# test_select()
test_linearlayer()