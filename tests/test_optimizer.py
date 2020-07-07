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

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag


def gen_dag1():
    #(a*b) + 4 - b
    x = Input(name="x")
    y = Input(name="y")
    z = Input(name="z")
    i = Mul(x, y, name="i")
    j = Sub(y, z, name="j")
    k = Mul(i, j, name="k")

    dag = Dag(outputs=[k], inputs=[x, y, z])
    return dag


def test_fig3():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a, b = Interval(-3, 2), Interval(4, 8)
    evaluator.eval(a=a, b=b)

    bfo = BitFlowOptimizer(evaluator, 'z', 8)
    bfo.solve()

    assert bfo.visitor.IBs == {'a': 4, 'b': 5, 'd': 7, 'c': 4, 'e': 6, 'z': 7}
    assert bfo.initial == 12


def test_dag1():
    dag1 = gen_dag1()
    evaluator = NumEval(dag1)

    x, y, z = 2, 5, 3
    evaluator.eval(x=x, y=y, z=z)

    bfo = BitFlowOptimizer(evaluator, 'k', 5)
    bfo.solve()

    print(bfo.visitor.IBs)
    print(bfo.initial)

    assert bfo.visitor.IBs == {'x': 3, 'y': 4, 'i': 5, 'z': 3, 'j': 3, 'k': 6}
    assert bfo.initial == 10


def test_print():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a, b = Interval(-3, 2), Interval(4, 8)
    evaluator.eval(a=a, b=b)

    bfo = BitFlowOptimizer(evaluator, 'z', 5)
    bfo.solve()

    print("\nRESULTS:")
    print("node, IB, FB ")
    for node in bfo.fb_sols.keys():
        print(f"{node}, {bfo.visitor.IBs[node]}, {bfo.fb_sols[node]}")
