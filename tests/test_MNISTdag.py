import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product


def gen_multiply():
    a = Input(name="a")
    b = Input(name="b")
    z = Mul(a,b,name="z")

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_dotproduct():
    a = Input(name="a")
    b = Input(name="b")
    z = dot_product(a, b)

    fig = Dag(outputs=[z], inputs=[a,b])
    return fig

def gen_relu():
    a = Input(name="a")
    b = Relu(a,name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig

def gen_reduce():
    a = Input(name="a")
    b = Reduce(a, name="b")

    fig = Dag(outputs=[b], inputs=[a])
    return fig

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

def test_Multiply():
    tmp = gen_multiply()
    evaluator = NumEval(tmp)
    a = torch.tensor([1,2])
    b = torch.tensor([3,4])
    res = evaluator.eval(a=a,b=b)
    gold = torch.tensor([3,8])
    assert torch.all(torch.eq(res,gold))
    return


def test_Relu():

        tmp = gen_relu()
        evaluator = NumEval(tmp)

        a = -1
        res = evaluator.eval(a=a)
        gold = 0
        assert res == gold

        a = 0
        res = evaluator.eval(a=a)
        gold = 0
        assert res == gold

        a = -1000
        res = evaluator.eval(a=a)
        gold = 0
        assert res == gold

        a = 1000
        res = evaluator.eval(a=a)
        gold = 1000
        assert res == gold

        a = 3.14
        res = evaluator.eval(a=a)
        gold = 3.14
        assert res == gold
        return


def test_Reduce():
    tmp = gen_reduce()
    evaluator = NumEval(tmp)

    a = torch.tensor([2,4,1,3])
    res = evaluator.eval(a=a)
    #print(res)

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
    return

test_Relu()
test_Reduce()
test_Multiply()
test_dotproduct()
