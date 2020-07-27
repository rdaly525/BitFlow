import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Len, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply

def gen_display():
    a = Input(name="a")
    z = a
    fig = Dag(outputs=[z], inputs=[a])
    return fig


def gen_multiply():
    a = Input(name="a")
    b = Input(name="b")
    z = Mul(a,b,name="z")

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_length():
    a = Input(name="a")
    z = Len(a)

    fig = Dag(outputs=[z], inputs=[a])
    return fig

def gen_dotproduct():
    a = Input(name="a")
    b = Input(name="b")
    z = dot_product(a, b)

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_matrix_multiply():
    a = Input(name="a")
    b = Input(name="b")

    z = matrix_multiply(a, b)

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_matrix_mul(row,col):
    a = Input(name="a")
    b = Input(name="b")

    z = matrix_multiply(a, b,row,col)

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_relu():
    a = Input(name="a")
    b = Relu(a,name="b")

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
    b = a[0:3:2]

    fig = Dag(outputs=[b], inputs=[a])
    return fig

def gen_reduce_z():
    a = Input(name="a")
    b = Reduce(a, 2, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig


def test_length():
    tmp = gen_length()
    evaluator = NumEval(tmp)
    a = torch.tensor([1])
    res = evaluator.eval(a=a)
    gold = 1
    assert res==gold

    a = torch.tensor([1,2])
    res = evaluator.eval(a=a)
    gold = 2
    assert res == gold

    a = torch.tensor([1,3,4])
    res = evaluator.eval(a=a)
    gold = 3
    assert res == gold

    a = torch.tensor([[1,3,4],[1,2,3]])
    res = evaluator.eval(a=a)
    gold = 2
    assert res == gold

    a = torch.tensor([[1,3],[1,2],[1,2]])
    res = evaluator.eval(a=a)
    gold = 3
    assert res == gold

    a = torch.tensor([[1,3],[1,2],[1,2]])
    a = a[0]
    res = evaluator.eval(a=a)
    gold = 2
    assert res == gold

    a = torch.tensor([[[2,2]]])
    a = a[0][0]
    res = evaluator.eval(a=a)
    gold = 2
    assert res == gold

    return

def test_dotproduct():
    tmp = gen_dotproduct()
    evaluator = NumEval(tmp)
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([11])
    assert torch.all(torch.eq(res, gold))

    a = torch.tensor([1, 2,3,0,-1])
    b = torch.tensor([3, 4,-1,2.2,-1])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([9])
    assert torch.all(torch.eq(res, gold))

    a = torch.tensor([1])
    b = torch.tensor([3])
    res = evaluator.eval(a=a, b=b)
    gold = torch.tensor([3])
    assert torch.all(torch.eq(res, gold))

    return


def test_matrix_mult():

    tmp = gen_matrix_mul(2,2)
    evaluator = NumEval(tmp)

    a = torch.tensor([[1,2],[3,4]])
    b = torch.tensor([[5,6],[7,8]])

    res = evaluator.eval(a=a, b=b)
    print(res)
    print()

    tmp = gen_matrix_mul(2, 2)
    evaluator = NumEval(tmp)

    a = torch.tensor([[1, 0], [3, 4]])
    b = torch.tensor([[5, 6], [0, 8]])

    res = evaluator.eval(a=a, b=b)
    print(res)
    print()

    a = torch.tensor([[1, 2], [3, 4], [5,6]])
    b = torch.tensor([[1,2,3,4], [5,6,7,8]])

    tmp = gen_matrix_mul(3,4)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a,b=b)
    print(res)
    print()

    a = torch.tensor([[1,2,6,7], [3, 4, 5, 1], [5, 6, 1,2]])
    b = torch.tensor([[1, 2,6,7], [3, 4, 5, 1], [5, 6, 1,2], [3, 4, 5, 1]])
    tmp = gen_matrix_mul(3, 4)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a, b=b)
    print(res)
    print()


    a = torch.tensor([[1, 2, 6, 7], [3, 4, 5, 1], [5, 6, 1, 2], [3, 4, 5, 1]])
    b = torch.tensor([[1,2,4,7]])
    tmp = gen_matrix_mul(4, 1)
    evaluator = NumEval(tmp)
    res = evaluator.eval(a=a, b=b)
    print(res)

    return

def test_select():
    tmp = gen_select()
    evaluator = NumEval(tmp)
    a = torch.tensor([100,20000,20,400,22,11])
    res = evaluator.eval(a=a)

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

        a = torch.tensor([3.14,2.5,3.2])
        res = evaluator.eval(a=a)
        gold = torch.tensor([3.14,2.5,3.2])
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


def test_Reduce():
    tmp_x = gen_reduce_x()
    tmp_y = gen_reduce_y()
    tmp_z = gen_reduce_z()

    evaluator = NumEval(tmp_x)

    a = torch.tensor([2,4,1,3])
    res = evaluator.eval(a=a)

    gold = 10
    assert res == gold

    a = torch.tensor([1,2,3,9])
    res = evaluator.eval(a=a)
    gold = 15
    assert res == gold

    a = torch.tensor([1,2,3,9,-1,3.3,4.6, 12.3])
    res = evaluator.eval(a=a)
    gold = 34.2
    assert res == gold

    a = torch.tensor([112.3])
    res = evaluator.eval(a=a)
    gold = 112.3
    assert res == gold

    a = torch.tensor([[112,11],[1,10]])
    res = evaluator.eval(a=a)
    gold = torch.tensor([113,  21])
    assert torch.all(torch.eq(res, gold))

    evaluator_y = NumEval(tmp_y)

    a = torch.tensor([[112, 11], [1, 10]])
    res = evaluator_y.eval(a=a)
    gold = torch.tensor([123,  11])
    assert torch.all(torch.eq(res, gold))

    evaluator_z = NumEval(tmp_z)

    a = torch.tensor([[[11,112], [10,11]], [[1,2], [3,10]]])
    res = evaluator_z.eval(a=a)
    gold = torch.tensor([[123,21],[3,13]])
    assert torch.all(torch.eq(res, gold))

    return

test_Relu()
test_Reduce()
test_dotproduct()
test_select()
test_length()
test_matrix_mult()
