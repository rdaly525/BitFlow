from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
import torch

def gen_fig3():
    #(a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a,b])
    return fig3_dag

#Evaluate it in the context of simple values
def test_fig3_integers():
    fig3 = gen_fig3()
    evaluator = NumEval(fig3)

    a, b = 3, 5
    res = evaluator.eval(a=a, b=b)
    gold = 14
    assert res == gold

#Evaluate it in the context of Intervals
def test_fig3_IA():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a = Interval(0, 5)
    b = Interval(3, 8)
    res = evaluator.eval(a=a, b=b)
    gold = Interval(-4, 41)
    assert res == gold

#Evaluate it in the context of Torch
def test_fig3_IA():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a = torch.Tensor([3])
    b = torch.Tensor([5])
    res = evaluator.eval(a=a, b=b)
    gold = torch.Tensor([14])
    assert res == gold

