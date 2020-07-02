from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch

def gen_rounded_fig3():
    # 1. Define the DAG and weight nodes
    W = Input(name="W")
    X = Input(name="X")

    a = X[0]
    b = X[1]
    c = Constant(4, name="c")

    # 2. Add Round Nodes after +/-/* in DAG
    d = Round(Mul(a, b, name="d"), W[0])
    e = Round(Add(d, c, name="e"), W[1])
    z = Round(Sub(e, b, name="z"), W[2])

    fig3 = Dag(outputs=[z], inputs=[X,W])

    return fig3

def test_fig3():
    dag = gen_rounded_fig3()
    evaluator = TorchEval(dag)

    BitFlow(dag)
    return

test_fig3()
