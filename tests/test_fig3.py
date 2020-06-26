from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval

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


class NodePrinter(Visitor):
    def __init__(self, node_values):
        self.node_values = node_values


    #Visitor method for only 'Input' nodes
    def visit_Input(self, node: Input):
        #This method is only called on Input nodes
        assert isinstance(node, Input)

        #Inputs have no children, so no need to call Visitor.generic_visit
        # (Although no harm in doing so)
        assert len(list(node.children())) == 0

        #I now have access to the node and anything I initialized this class with
        print(f"Input Node {node} has a value of {self.node_values[node]}")


    #Generic visitor method for a node
    #This method will be run on each node unless 
    # a visit_<NodeType> method is defined 
    def generic_visit(self, node: DagNode):
        #Call this to visit node's children first
        Visitor.generic_visit(self, node)

        print(f"Generic Node {node} has a value of {self.node_values[node]} and children values")
        for child_node in node.children():
            print(f"  {child_node}:  {self.node_values[child_node]}")


    #I could also define a custom visitor for Add
    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)
        assert isinstance(node, Add)
        print(f"Add Node {node} has a value of {self.node_values[node]}")
        for child_node in node.children():
            print(f"  {child_node}:  {self.node_values[child_node]}")


def test_print():
    fig3 = gen_fig3()
    evaluator = NumEval(fig3)

    a, b = 3, 5
    evaluator.eval(a=a, b=b)
    node_values = evaluator.node_values
    node_printer = NodePrinter(node_values)

    # Visitor classes have a method called 'run' that takes in a dag and runs all the 
    # visit methods on each node
    node_printer.run(fig3)
