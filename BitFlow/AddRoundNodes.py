from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Round, Output, Select, Concat, Reduce
from DagVisitor import Visitor, Transformer, AbstractDag
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval, AbstractEval
import torch

from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Select
from DagVisitor import Visitor
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil
from .Precision import PrecisionNode
from scipy.optimize import fsolve, minimize, basinhopping

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

    def __init__(self, P, R, O):
        self.P = P
        self.R = R
        self.O = O
        self.area_weight = 0
        self.round_count = 0
        self.input_count = 0
        self.output_count = 0
        self.range_count = 0
        self.rounded_outputs = []
        self.allroots = []
        self.order = []

    def doit(self, dag: Dag):  # takes a Dag and returns new Dag with round nodes added in

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

        self.area_weight += 1
        self.order.append(node.name)

        if isinstance(node, LookupTable):
            self.area_weight += 9

        # if isinstance(node, LookupTable):
        #     return None

        if isinstance(node, Input):
            # current node + need to get prec_input
            returnNode = Round(node, Select(
                self.P, self.round_count), Select(self.R, self.range_count), name=node.name + "_round")

            self.input_count += 1
            self.round_count += 1
            self.range_count += 1

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
            returnNode = Round(node, Select(self.P, self.round_count), Select(self.R, self.range_count),
                               name=node.name + "_round_P")
            self.round_count += 1
            self.range_count += 1
            return returnNode

# class NodePrinter(Visitor):
#     def __init__(self, node_values):
#         self.node_values = node_values
#
#     def test_print(self):
#         fig3 = gen_fig3()
#         evaluator = NumEval(fig3)
#
#         a, b = 3, 5
#         evaluator.eval(a=a, b=b)
#         node_values = evaluator.node_values
#         node_printer = NodePrinter(node_values)
#
#         # Visitor classes have a method called 'run' that takes in a dag and runs all the
#         # visit methods on each node
#         node_printer.run(fig3)
#
#     # Generic visitor method for a node
#     # This method will be run on each node unless
#     # a visit_<NodeType> method is defined
#     def generic_visit(self, node: DagNode):
#         # Call this to visit node's children first
#         Visitor.generic_visit(self, node)
#
#         print(
#             f"Generic Node {node} has a value of {self.node_values[node]} and children values")
#         for child_node in node.children():
#             print(f"  {child_node}:  {self.node_values[child_node]}")
#
#
# class AddRoundNodes(Transformer):
#
#     def __init__(self, W, O):
#         self.W = W
#         # self.X = X
#         self.O = O
#         self.round_count = 0
#         self.input_count = 0
#         self.output_count = 0
#         self.rounded_outputs = []
#         self.allroots = []
#
#     def doit(self, dag: Dag):  # takes a Dag and returns new Dag with round nodes added in
#
#         new_inputs = dag.inputs
#
#         self.allroots = list(dag.roots())
#
#         self.run(dag)
#
#         new_inputs.append(self.W)
#         # new_inputs.append(self.X)
#         new_inputs.append(self.O)
#
#         new_outputs = list(self.rounded_outputs)
#
#         # return dag that has taken precision as an input
#         return Dag(outputs=new_outputs, inputs=new_inputs)
#
#     def generic_visit(self, node: DagNode):
#
#         if isinstance(node, Output):
#             return None
#
#         if isinstance(node, Select):
#             return None
#
#         if isinstance(node, Concat):
#             return None
#
#         # make sure code run on all children nodes first
#         Transformer.generic_visit(self, node)
#
#         if isinstance(node, Input):
#             # current node + need to get prec_input
#             returnNode = Round(node, Select(
#                 self.W, self.round_count), name=node.name + "_round")
#             self.input_count += 1
#             self.round_count += 1
#
#         else:
#             if (node in self.allroots):
#
#                 returnNode = Round(node, Select(self.O, self.output_count),
#                                    name=node.name + "_round")
#                 self.rounded_outputs.append(returnNode)
#                 self.output_count += 1
#
#             else:
#                 returnNode = Round(node, Select(self.W, self.round_count),
#                                    name=node.name + "_round_W")
#                 self.round_count += 1
#         return returnNode
#
#
# class AllKeys(Visitor):
#
#     def __init__(self):
#         # self.new_dict = {new_list: 0 for new_list in range(3206)}
#         # self.i = 0
#         self.listV = []
#
#     def doit(self, dag: Dag):  # takes a Dag and returns new Dag with round nodes added in
#
#         self.run(dag)
#         return self.gen_dictionary()
#
#
#     def generic_visit(self, node: DagNode):
#         # Call this to visit node's children first
#         Visitor.generic_visit(self, node)
#         # self.new_dict[node.name]=10
#         # self.i +=1
#         self.listV.append(node.name)
#         #print(self.listV)
#         # print(self.new_dict)
#
#     def gen_dictionary(self):
#         nodeD = dict.fromkeys(self.listV, 10)
#         return nodeD
#
#         # print(f"{node}: {type(node)}")
#         # for child_node in node.children():
#         #     print(f"  {child_node}")
#
#
# class NodePrinter1(Visitor):
#
#     def generic_visit(self, node: DagNode):
#         # Call this to visit node's children first
#         Visitor.generic_visit(self, node)
#
#         print(f"{node}: {type(node)}")
#         for child_node in node.children():
#             print(f"  {child_node}")
#
#
# class BitFlowVisitor(Visitor):
#
#     def __init__(self, node_values):
#         self.node_values = node_values
#         self.errors = {}
#         self.IBs = {}
#         self.area_fn = ""
#         self.concat = 0
#         self.reduce = 0
#         self.concatAdd = 0
#
#     def handleIB(self, node):
#         ib = 0
#         x = self.node_values[node]
#
#         x = torch.mean(x)
#         if isinstance(x, Interval):
#             alpha = 2 if (log2(abs(x.hi)).is_integer()) else 1
#             ib = ceil(log2(max(abs(x.lo), abs(x.hi)))) + alpha
#         else:
#             alpha = 2 if (log2(abs(x)).is_integer()) else 1
#             ib = ceil(log2(abs(x))) + alpha
#         self.IBs[node.name] = int(ib)
#
#     def getChildren(self, node):
#         children = []
#         for child_node in node.children():
#             children.append(child_node)
#
#         return (children[0], children[1])
#
#     def getChildren_Conc(self, node):
#         children = []
#         for child_node in node.children():
#             children.append(child_node)
#
#         return children
#
#     def getChild(self, node):
#         children = []
#         for child_node in node.children():
#             children.append(child_node)
#
#         return (children[0])
#
#     def visit_Input(self, node: Input):
#         self.handleIB(node)
#
#         val = 0
#         if isinstance(self.node_values[node], Interval):
#             x = self.node_values[node]
#             val = max(abs(x.lo), abs(x.hi))
#         else:
#             val = self.node_values[node]
#
#         self.errors[node.name] = PrecisionNode(val, node.name, [])
#
#         print("exit input")
#
#     def visit_Select(self, node: Select):
#
#         self.handleIB(node)
#         val = self.node_values[node]
#         val = torch.mean(val)
#
#         self.errors[node.name] = PrecisionNode(val, node.name, [])
#         print(self.errors[node.name])
#
#         print("exit select")
#
#     def visit_Constant(self, node: Constant):
#         self.handleIB(node)
#
#         val = self.node_values[node]
#         val = torch.mean(val)
#         self.errors[node.name] = PrecisionNode(val, node.name, [])
#         print("exit constant")
#
#     # def visit_Reduce(self, node: Reduce):
#     #     Visitor.generic_visit(self, node)
#     #
#     #     # print("AREA REDUCE VISITED")
#     #     self.reduce += 1
#     #     # print(self.reduce)
#     #
#     #     lhs = self.getChild(node)
#     #     self.errors[node.name] = self.errors[lhs.name]
#     #     # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#     #     self.area_fn += f"+max({lhs.name})"
#     #
#     #     # print(self.area_fn)
#     #     print("exit reduce")
#
#     # def visit_Concat(self, node: Concat):
#     #
#     #     Visitor.generic_visit(self, node)
#     #
#     #     # print("AREA CONCAT VISITED")
#     #     self.concat += 1
#     #     # print(self.concat)
#     #
#     #     listChild = self.getChildren_Conc(node)
#     #     print(listChild)
#     #     lhs = self.getChild(node)
#     #     self.errors[node.name] = self.errors[lhs.name]
#     #     # print("children vector length",len(listChild))
#     #     #self.errors[node.name] = self.errors[listChild]
#     #     # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#     #
#     #     self.concatAdd += len(listChild)
#     #     # print("concatAdd", self.concatAdd)
#     #     self.area_fn += f"+max({listChild})"
#     #     # print(self.area_fn)
#     #
#     #     # print(self.area_fn)
#     #     print("exit concat")
#
#     def visit_Add(self, node: Add):
#         Visitor.generic_visit(self, node)
#
#         self.handleIB(node)
#
#         lhs, rhs = self.getChildren(node)
#         print("hello:",lhs,rhs)
#         print(self.errors[rhs.name])
#         self.errors[node.name] = self.errors[lhs.name].add(
#             self.errors[rhs.name], node.name)
#         # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#         self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#
#         print("exit add")
#
#
#     def visit_Output(self, node: Output):
#         print("OUTPUT")
#         assert 1 ==0
#
#     def visit_Sub(self, node: Sub):
#         Visitor.generic_visit(self, node)
#
#         self.handleIB(node)
#         lhs, rhs = self.getChildren(node)
#         self.errors[node.name] = self.errors[lhs.name].sub(
#             self.errors[rhs.name], node.name)
#         # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#         self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#
#         print("exit subtract")
#
#     def visit_Mul(self, node: Mul):
#         Visitor.generic_visit(self, node)
#
#         self.handleIB(node)
#         print("node", node)
#         lhs, rhs = self.getChildren(node)
#         print(node.name)
#         print("left",lhs.name)
#         self.errors[node.name] = self.errors[lhs.name].mul(
#             self.errors[rhs.name], node.name)
#         self.area_fn += f"+1 * ({self.IBs[lhs.name]} + {lhs.name})*({self.IBs[rhs.name]} + {rhs.name})"
#
#         print("exit mul")
#
#
#     def visit_Reduce(self, node: Reduce):
#         Visitor.generic_visit(self, node)
#
#         self.handleIB(node)
#         lhs= self.getChild(node)
#         self.errors[node.name] = self.errors[lhs.name]
#         # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#         self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name})"
#
#         print("exit reduce")
#
#     def visit_Concat(self, node: Concat):
#
#             Visitor.generic_visit(self, node)
#             print(node)
#             self.handleIB(node)
#             lhs, rhs = self.getChildren(node)
#             self.errors[node.name] = self.errors[lhs.name].add(
#                 self.errors[rhs.name], node.name)
#             # self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#             self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
#
#             print("exit concat")
#
# class BitFlowOptimizer():
#     def __init__(self, evaluator, outputs):
#
#         node_values = evaluator.node_values
#         visitor = BitFlowVisitor(node_values)
#         visitor.run(evaluator.dag)
#
#         self.visitor = visitor
#         self.error_fn = ""
#         self.ufb_fn = ""
#
#         for output in outputs:
#
#             print(outputs,output)
#             self.error_fn += f"+2**(-{outputs[output]}-1) - (" + \
#                              visitor.errors[output].getExecutableError() + ")"
#             #print(self.error_fn)
#             self.ufb_fn += visitor.errors[output].getExecutableUFB()
#         self.area_fn = visitor.area_fn[1:]
#
#         self.outputs = outputs
#
#
#
#         vars = list(visitor.node_values)
#         for (i, var) in enumerate(vars):
#             vars[i] = var.name
#         self.vars = vars
#
#     def calculateInitialValues(self):
#         # print("CALCULATING INITIAL VALUES USING UFB METHOD...")
#         # bnd = f"{-2**(-self.output_precision-1)} == 0"
#         bnd = ""
#         for output in self.outputs:
#             bnd += f"{-2 ** (-self.outputs[output] - 1)}"
#         self.ufb_fn += bnd
#         # print(f"UFB EQ: {self.ufb_fn}")
#         # print(f"-----------")
#
#         exec(f'''def UFBOptimizerFn(UFB):
#              return  {self.ufb_fn}''', globals())
#
#         sol = ceil(fsolve(UFBOptimizerFn, 0.01))
#         self.initial = sol
#
#         # m = GEKKO()
#         # UFB = m.Var(value=0,integer=True)
#         # m.options.IMODE=2
#         # m.options.SOLVER=3
#         #
#         # exec(f'''def UFBOptimizerFn(UFB):
#         #     return  {self.ufb_fn}''', globals())
#         #
#         # m.Equation(UFBOptimizerFn(UFB))
#         # m.solve(disp=True)
#         #
#         # sol = ceil(UFB.value[0])
#         # self.initial = sol
#         # print(f"UFB = {sol}\n")
#
#     def solve(self):
#         self.calculateInitialValues()
#         print("SOLVING AREA/ERROR...")
#         # self.error_fn = f"2**(-{self.output_precision}-1)>=" + self.error_fn
#
#         print(f"ERROR EQ: {self.error_fn}")
#         print(f"AREA EQ: {self.area_fn}")
#         print(f"-----------")
#
#         filtered_vars = []
#         for var in self.vars:
#             if var not in self.outputs:
#                 filtered_vars.append(var)
#
#         exec(f'''def ErrorConstraintFn(x):
#              {','.join(filtered_vars)} = x
#              return  {self.error_fn}''', globals())
#
#         exec(f'''def AreaOptimizerFn(x):
#              {','.join(filtered_vars)} = x
#              return  {self.area_fn}''', globals())
#
#         x0 = [self.initial for i in range(len(filtered_vars))]
#         bounds = [(0, 64) for i in range(len(filtered_vars))]
#
#         con = {'type': 'ineq', 'fun': ErrorConstraintFn}
#
#         # note: minimize uses SLSQP by default but I specify it to be explicit; we're using basinhopping to find the global minimum while using SLSQP to find local minima
#         minimizer_kwargs = {'constraints': (
#             [con]), 'bounds': bounds, 'method': "SLSQP"}
#         solution = basinhopping(AreaOptimizerFn, x0,
#                                 minimizer_kwargs=minimizer_kwargs)
#
#         sols = dict(zip(filtered_vars, solution.x))
#
#         for key in sols:
#             sols[key] = ceil(sols[key])
#             print(f"{key}: {sols[key]}")

        # self.fb_sols = sols
