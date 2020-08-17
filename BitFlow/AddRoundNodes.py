from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Round, Output, Select, Concat, Reduce
from DagVisitor import Visitor, Transformer, AbstractDag
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval, AbstractEval
import torch


class NodePrinter(Visitor):
    def __init__(self, node_values):
        self.node_values = node_values

    def test_print(self):
        fig3 = gen_fig3()
        evaluator = NumEval(fig3)

        a, b = 3, 5
        evaluator.eval(a=a, b=b)
        node_values = evaluator.node_values
        node_printer = NodePrinter(node_values)

        # Visitor classes have a method called 'run' that takes in a dag and runs all the
        # visit methods on each node
        node_printer.run(fig3)

    # Generic visitor method for a node
    # This method will be run on each node unless
    # a visit_<NodeType> method is defined
    def generic_visit(self, node: DagNode):
        # Call this to visit node's children first
        Visitor.generic_visit(self, node)

        print(
            f"Generic Node {node} has a value of {self.node_values[node]} and children values")
        for child_node in node.children():
            print(f"  {child_node}:  {self.node_values[child_node]}")


class AddRoundNodes(Transformer):


    def __init__(self, W, O):
        self.W = W
        # self.X = X
        self.O = O
        self.round_count = 0
        self.input_count = 0
        self.output_count = 0
        self.rounded_outputs = []
        self.allroots = []
        self.counterA = 0
        self.counterM = 0

    def doit(self, dag: Dag):  # takes a Dag and returns new Dag with round nodes added in

        new_inputs = dag.inputs

        self.allroots = list(dag.roots())

        self.run(dag)

        new_inputs.append(self.W)
        # new_inputs.append(self.X)
        new_inputs.append(self.O)

        new_outputs = list(self.rounded_outputs)

        # return dag that has taken precision as an input
        return Dag(outputs=new_outputs, inputs=new_inputs)

    def generic_visit(self, node: DagNode):



        if isinstance(node, Add):
            self.counterA+=1
            print("entered1")

        if isinstance(node,Mul):
            self.counterM+=1
            print("entered2")

        if isinstance(node, Output):
            return None

        if isinstance(node, Select):
            return None

        if isinstance(node, Concat):
            return None

        # make sure code run on all children nodes first
        Transformer.generic_visit(self, node)

        if isinstance(node, Input):
            # current node + need to get prec_input
            returnNode = Round(node, Select(
                self.W, self.round_count), name=node.name + "_round")
            self.input_count += 1
            self.round_count += 1

        else:
            if (node in self.allroots):

                returnNode = Round(node, Select(self.O, self.output_count),
                                   name=node.name + "_round")
                self.rounded_outputs.append(returnNode)
                self.output_count += 1

            else:
                returnNode = Round(node, Select(self.W, self.round_count),
                                   name=node.name + "_round_W")
                self.round_count += 1
        return returnNode


class NodePrinter1(Visitor):

    # a visit_<NodeType> method is defined
    def generic_visit(self, node: DagNode):
        # Call this to visit node's children first
        Visitor.generic_visit(self, node)

        print(f"{node}: {type(node)}")
        for child_node in node.children():
            print(f"  {child_node}")
class MNIST_area(Visitor):

    def __init__(self,evaluator):

        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        self.visitor = visitor
        self.error_fn = ""
        self.ufb_fn = ""

        #TODO: Implement for Outputs

        self.area_fn = visitor.area_fn[1:]

        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        self.vars = vars


class BitFlowVisitor(Visitor):

    def __init__(self, node_values):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}
        self.area_fn = ""
        self.concat = 0
        self.reduce = 0

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)

        return (children[0], children[1])

    def getChildren_Conc(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)

        return children


    def getChild(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)

        return (children[0])

    # def visit_Input(self, node: Input):
    #
    #
    #     val = self.node_values[node]
    #
    #     #self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Select(self, node: Select):
        Visitor.generic_visit(self, node)

    # def visit_Constant(self, node: Constant):
    #
    #     val = self.node_values[node]
    #     #self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Reduce(self, node: Reduce):
        Visitor.generic_visit(self, node)

        print("AREA REDUCE VISITED")
        self.reduce += 1
        print(self.reduce)

        lhs = self.getChild(node)
        # self.errors[node.name] = self.errors[lhs.name].add(
        #     self.errors[rhs.name], node.name)
        # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        self.area_fn += f"+max({lhs.name})"


        #print(self.area_fn)

    def visit_Concat(self, node: Concat):


        Visitor.generic_visit(self, node)

        print("AREA CONCAT VISITED")
        self.concat += 1
        print(self.concat)

        listChild = self.getChildren_Conc(node)
        print("children vector length",len(listChild))
        # self.errors[node.name] = self.errors[lhs.name].add(
        #     self.errors[rhs.name], node.name)
        # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"


        self.area_fn += f"+max({listChild})"
        print(self.area_fn)


        #print(self.area_fn)

    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)

        lhs, rhs = self.getChildren(node)
        # self.errors[node.name] = self.errors[lhs.name].add(
        #     self.errors[rhs.name], node.name)
        # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        self.area_fn += f"+max({lhs.name}, {rhs.name})"

        print("AREA ADD VISITED")
        print(self.area_fn)

    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)


        lhs, rhs = self.getChildren(node)
        # self.errors[node.name] = self.errors[lhs.name].sub(
        #     self.errors[rhs.name], node.name)
        # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        self.area_fn += f"+max({lhs.name}, {rhs.name})"

    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)


        lhs, rhs = self.getChildren(node)
        # self.errors[node.name] = self.errors[lhs.name].mul(
        #     self.errors[rhs.name], node.name)
        self.area_fn += f"+1 * ({lhs.name})*({rhs.name})"
        print(self.area_fn)


    # # a visit_<NodeType> method is defined
    # def generic_visit(self, node: DagNode):
    #     # Call this to visit node's children first
    #     Visitor.generic_visit(self, node)
    #
    #     print(f"{node}: {type(node)}")
    #     for child_node in node.children():
    #         print(f"  {child_node}")
