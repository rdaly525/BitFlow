from curis.node import Input, Constant, Dag, Add, Sub, Mul
from curis.visitor import Visitor
from curis.IA import Interval

def gen_fig3():
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(output=z, inputs=[a,b])
    return fig3_dag

def test_printing():
    class Printer(Visitor):
        def __init__(self, dag):
            self.res = "\n"
            super().__init__(dag)

        def generic_visit(self, node):
            child_names = ", ".join([str(child.name) for child in node.children()])
            self.res += f"{node.name}({child_names}) : {node.kind()[0]}\n"
            Visitor.generic_visit(self, node)

    fig3 = gen_fig3()
    p = Printer(fig3)
    print(p.res)

#Evaluate it in the context of simple values
def test_fig3_integers():
    fig3 = gen_fig3()

    a, b = 3, 5
    assert fig3.eval(a, b) == 7

#Evaluate it in the context of Intervals
def test_fig3_IA():
    fig3 = gen_fig3()

    a = Interval(0, 5)
    b = Interval(3, 8)
    assert fig3.eval(a, b) == Interval(-1, 14)

def test_fig3_torch():
    #TODO
    fig3 = gen_fig3()


