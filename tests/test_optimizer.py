from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Optimization import BitFlowOptimizer

def gen_fig3():
    #(a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(output=z, inputs=[a,b])
    return fig3_dag

def test_fig3():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a, b = Interval(-3, 2), Interval(4, 8)
    evaluator.eval(a=a, b=b)

    bfo = BitFlowOptimizer(evaluator, 'z', 8)
    bfo.solve()

    assert bfo.visitor.IBs == {'a': 4, 'b': 5, 'd': 7, 'c': 4, 'e': 6, 'z': 7}
    assert bfo.initial == 12
