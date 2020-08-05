from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Round, Output, Select
from DagVisitor import Visitor, Transformer, AbstractDag
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval, AbstractEval
import torch


class AddRoundNodes(Transformer):

    def __init__(self, P, R, O):
        self.P = P
        self.R = R
        self.O = O
        self.round_count = 0
        self.input_count = 0
        self.output_count = 0
        self.range_count = 0
        self.rounded_outputs = []
        self.allroots = []

    def doit(self, dag: Dag):  # takes a Dag and returns new Dag with round nodes added in

        new_inputs = dag.inputs

        self.allroots = list(dag.roots())
        self.run(dag)

        new_inputs.append(self.P)
        new_inputs.append(self.R)
        new_inputs.append(self.O)

        #new_outputs = list(self.rounded_outputs)

        # return dag that has taken precision as an input
        return Dag(outputs=dag.outputs, inputs=new_inputs)

    def generic_visit(self, node: DagNode):

        if isinstance(node, Output):
            return None

        # make sure code run on all children nodes first
        Transformer.generic_visit(self, node)

        # for child in node.children():
        #     assert isinstance(child, Round)

        if isinstance(node, Input):
            # current node + need to get prec_input
            returnNode = Round(node, Select(
                self.P, self.round_count), Select(self.R, self.range_count), name=node.name + "_round")
            self.input_count += 1
            self.round_count += 1
            self.range_count += 1

        else:
            if (node in self.allroots):

                return Select(self.O, self.output_count)

                # returnNode = Round(node, Select(self.O, self.output_count), Select(self.R, self.range_count),
                #                    name=node.name + "_round")
                # self.rounded_outputs.append(returnNode)
                self.output_count += 1
                # self.range_count += 1

            else:
                returnNode = Round(node, Select(self.P, self.round_count), Select(self.R, self.range_count),
                                   name=node.name + "_round_W")

                self.round_count += 1
                self.range_count += 1
                return returnNode
