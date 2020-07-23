from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Round, Output, Select
from DagVisitor import Visitor, Transformer, AbstractDag
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval, AbstractEval
import torch


class AddRoundNodes(Transformer):

    def __init__(self, W, O):
        self.W = W
        #self.X = X
        self.O = O
        self.round_count = 0
        self.input_count = 0
        self.output_count = 0
        self.rounded_outputs = []
        self.allroots = []

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

        if isinstance(node, Output):
            return None

        # make sure code run on all children nodes first
        Transformer.generic_visit(self, node)

        for child in node.children():
            assert isinstance(child, Round)

        if isinstance(node, Input):
            # current node + need to get prec_input
            returnNode = Round(node, Select(
                self.W, self.round_count), name=node.name + "_round")
            self.input_count += 1
            self.round_count += 1

        else:
            if(node in self.allroots):

                returnNode = Round(node, Select(self.O, self.output_count),
                                   name=node.name + "_round")
                self.rounded_outputs.append(returnNode)
                self.output_count += 1

            else:
                returnNode = Round(node, Select(self.W, self.round_count),
                                   name=node.name + "_round_W")
                self.round_count += 1
        return returnNode
