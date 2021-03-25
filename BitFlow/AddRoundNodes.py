from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Round, Output, Select, LookupTable, BitShift
from DagVisitor import Visitor, Transformer, AbstractDag
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval, AbstractEval
from BitFlow.utils import LUTGenerator
import torch


class LookupTableTransformer(Transformer):
    def __init__(self, P, R, order):
        self.P = P
        self.R = R
        self.order = order

    def doit(self, dag: Dag):
        self.run(dag)
        return Dag(outputs=dag.outputs, inputs=dag.inputs)

    def generic_visit(self, node: DagNode):
        Transformer.generic_visit(self, node)
        if isinstance(node, LookupTable):
            return node
            child = node.child
            child_index = self.order.index(child.name)
            child_precision = int(self.P[child_index])
            child_range = int(self.R[child_index])

            min_val = -1 * (2 ** (child_precision + child_range - 1)) * \
                2 ** -child_precision
            max_val = (2 ** (child_precision + child_range - 1) - 1) * \
                2 ** -child_precision

            num_entries = 2 ** (child_precision + child_range)
            print(
                f"LOOKUP TABLE: [{min_val}, {max_val}] with {num_entries} entries")
            lut = LUTGenerator(
                node.func, [min_val, max_val], numel=num_entries)
            node.lut = lut
            return node

            #  (2 ** ({input_signal.name} + {input_signal.name}_ib)) * ({node.name} + {node.name}_ib)
        else:
            return node


class AddRoundNodes(Transformer):

    def __init__(self, P, R, O, area_map={}):
        self.P = P
        self.R = R
        self.O = O
        self.area_map = area_map
        self.area_weight = 0
        self.num_nodes = 0
        self.round_count = 0
        self.input_count = 0
        self.output_count = 0
        self.range_count = 0
        self.rounded_outputs = []
        self.allroots = []
        self.order = []

    # takes a Dag and returns new Dag with round nodes added in
    def doit(self, dag: Dag):

        new_inputs = dag.inputs

        self.allroots = list(dag.roots())
        self.run(dag)

        new_inputs.append(self.P)
        new_inputs.append(self.R)
        new_inputs.append(self.O)

        # return dag that has taken precision as an input
        return Dag(outputs=dag.outputs, inputs=new_inputs)

    def generic_visit(self, node: DagNode):
        # make sure code run on all children nodes first
        Transformer.generic_visit(self, node)
        self.num_nodes += 1

        nodetype = type(node).__name__
        if nodetype in self.area_map:
            self.area_weight += self.area_map[nodetype]
        else:
            self.area_weight += 1

        # if isinstance(node, LookupTable):
        #     return None

        if isinstance(node, Input):
            # current node + need to get prec_input
            returnNode = Round(node, Select(
                self.P, self.round_count), Select(self.R, self.range_count), name=node.name + "_round")

            self.input_count += 1
            self.round_count += 1
            self.range_count += 1
            self.order.append(node.name)

            return returnNode

        if isinstance(node, BitShift):
            self.round_count += 1
            self.range_count += 1
            return Round(node, 0., 0., name=node.name + "_round")

        # elif isinstance(node, LookupTable):

        #     self.round_count += 1
        #     self.range_count += 1

        #     return node

        elif (node in self.allroots):

            self.output_count += 1
            return Select(self.O, self.output_count)

        else:
            returnNode = Round(node, Select(self.P, self.round_count), Select(
                self.R, self.range_count), name=node.name + "_round")
            self.round_count += 1
            self.range_count += 1
            self.order.append(node.name)
            return returnNode
