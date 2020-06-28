from node import Input, Constant, Dag, Add, Sub, Mul, DagNode
from DagVisitor import Visitor
from IA import Interval
from Eval.IAEval import IAEval
from Eval.NumEval import NumEval
from math import log2, ceil
from Precision import PrecisionNode

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

class BitDAGVisitor(Visitor):
    def __init__(self, node_values):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}

    def handleIB(self, node):
        ib = 0
        x = self.node_values[node]
        if isinstance(x, Interval):
            alpha = 2 if (log2(x.hi) % 1 == 0) else 1
            ib = ceil(log2(max(abs(x.lo), abs(x.hi)))) + alpha
        else:
            alpha = 2 if (log2(x) % 1 == 0) else 1
            ib = ceil(log2(x)) + alpha
        self.IBs[node.name] = int(ib)

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)

        return (children[0], children[1])

    #Visitor method for only 'Input' nodes
    def visit_Input(self, node: Input):
        self.handleIB(node)

        val = 0
        if isinstance(self.node_values[node], Interval):
            x = self.node_values[node]
            val = max(abs(x.lo), abs(x.hi))
        else:
            val = self.node_values[node]

        self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Constant(self, node: Constant):
        self.handleIB(node)

        val = self.node_values[node]
        self.errors[node.name] = PrecisionNode(val, node.name, [])

    #I could also define a custom visitor for Add
    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].add(self.errors[rhs.name], node.name)


    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].sub(self.errors[rhs.name], node.name)


    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].mul(self.errors[rhs.name], node.name)


def test_print():
    fig3 = gen_fig3()
    evaluator = NumEval(fig3)

    a, b = 2, 3
    evaluator.eval(a=a, b=b)
    node_values = evaluator.node_values
    node_printer = BitDAGVisitor(node_values)

    # Visitor classes have a method called 'run' that takes in a dag and runs all the
    # visit methods on each node
    node_printer.run(fig3)
    print(node_printer.errors["z"])

test_print()
