from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Select, LookupTable, BitShift, Concat, Reduce, Output
from DagVisitor import Visitor
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil
from .Precision import PrecisionNode
from scipy.optimize import fsolve, minimize, basinhopping

import torch
import copy

from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Select, LookupTable, BitShift, Concat, Reduce
from DagVisitor import Visitor
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil
from .Precision import PrecisionNode
from scipy.optimize import fsolve, minimize, basinhopping

import torch
import copy

import sys
sys.setrecursionlimit(10000)


class BitFlowVisitor(Visitor):
    def __init__(self, node_values, calculate_IB=True):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}
        self.intervals = {}
        self.area_fn = ""
        self.train_MNIST = True
        if self.train_MNIST:
            self.calculate_IB = False

        #self.calculate_IB = calculate_IB

    def handleIB(self, node):
        if self.calculate_IB:
            ib = 0
            x = self.node_values[node]
            # print(f"{node}: {x}")
            if isinstance(x, Interval):
                alpha = 2 if (log2(abs(x.hi)).is_integer()) else 1
                ib = ceil(log2(max(abs(x.lo), abs(x.hi)))) + alpha
            else:
                if (x == 0.):
                    self.IBs[node.name] = 1
                elif (x < 1.):
                    self.IBs[node.name] = 0
                elif isinstance(x, list):
                    for (ind, val) in enumerate(x):
                        alpha = 2 if (log2(abs(val)).is_integer()) else 1
                        ib = ceil(log2(abs(val))) + alpha
                        self.IBs[f"{node.name}_getitem_{ind}"] = ib
                    return
                else:
                    alpha = 2 if (log2(abs(x)).is_integer()) else 1
                    ib = ceil(log2(abs(x))) + alpha
                    self.IBs[node.name] = int(ib)

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)
        if len(children) == 1:
            return children[0]
        return children

    def visit_Input(self, node: Input):
        # print("INPUT", node)
        self.handleIB(node)

        if node.name == 'X':
            # print("X---")
            error_mat = []
            for row in range(1):
                error_mat.append([])
                for col in range(784):
                    error_mat[row].append(PrecisionNode(
                        1., f"{node.name}", []))

                    # print(error_mat[row])

            self.errors[node.name] = error_mat
            # print(node.name)
            # print(len(error_mat[0]))

            return

        if node.name == 'weight':
            # print("WEIGHT---")
            error_mat = []
            for row in range(784):
                error_mat.append([])
                for col in range(10):
                    error_mat[row].append(PrecisionNode(
                        1., f"{node.name}", []))

            self.errors[node.name] = error_mat
            return

        # if node.name == 'bias':
        #     print("BIAS---")
        #     error_mat = []
        #     for row in range(1):
        #         error_mat.append([])
        #         for col in range(10):
        #             error_mat[row].append(PrecisionNode(
        #                 1., f"{node.name}", []))
        #     # print(node.name)
        #         self.errors[node.name] = error_mat
        #     return

        if node.name == 'bias':
            # print("BIAS---")
            error_mat = []
            for row in range(10):
                error_mat.append([])
                error_mat[row].append(PrecisionNode(
                        1., f"{node.name}", []))
            # print(node.name)
            self.errors[node.name] = error_mat

            # print(error_mat)
            # assert 0
            # return

        if self.calculate_IB:
            val = 0
            if isinstance(self.node_values[node], Interval):
                x = self.node_values[node]
                val = max(abs(x.lo), abs(x.hi))
            else:
                val = self.node_values[node]

            self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Constant(self, node: Constant):
        # print("CONSTANT", node)
        self.handleIB(node)

        if self.calculate_IB or self.train_MNIST:
            val = self.node_values[node]
            self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Add(self, node: Add):


        Visitor.generic_visit(self, node)
        # print("ADD", node)

        self.handleIB(node)

        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:


            error_mat = []

            for row in range(1):
                error_mat.append([])
                for col in range(10):

                 error_mat[row].append(self.errors[lhs.name][row][col].add(
                        self.errors[rhs.name][row][col][0], node.name))
            # print(node.name)
            self.errors[node.name] = error_mat
            # print(type(error_mat))



        if self.calculate_IB:
            self.area_fn += f"+1 * max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * max({lhs.name} + {lhs.name}, {rhs.name} + {rhs.name})"

    def visit_Sub(self, node: Sub):

        Visitor.generic_visit(self, node)
        # print("SUB", node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:
            self.errors[node.name] = self.errors[lhs.name].sub(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+1 * max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * max({lhs.name} + {lhs.name}, {rhs.name} + {rhs.name})"

    def visit_Mul(self, node: Mul):

        Visitor.generic_visit(self, node)
        # print("MUL", node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:
            res_list = []

            for i in range(0, len(self.errors[rhs.name])):
                res_list.append(self.errors[lhs.name][i].mul(self.errors[rhs.name][i],node.name))

            self.errors[node.name] = res_list

        if self.calculate_IB:
            self.area_fn += f"+1 * ({self.IBs[lhs.name]} + {lhs.name})*({self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * ({lhs.name} + {lhs.name})*({rhs.name} + {rhs.name})"

    def visit_BitShift(self, node: BitShift):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB:
            self.errors[node.name] = self.errors[lhs.name].mul(
                self.errors[rhs.name], node.name)

    def visit_Select(self, node: Select):

        Visitor.generic_visit(self, node)
        # print("SELECT", node)

        self.handleIB(node)
        input_signal = self.getChildren(node)
        #
        # print(node)

        if self.train_MNIST:
            if (isinstance(node.index, tuple)):

                self.errors[node.name] = self.errors[input_signal.name][0:][node.index[1]]
            elif node.name=='output0' or node.name=='output1' or node.name=='output2' or node.name=='output3' or node.name=='output4' or node.name=='output5' or node.name=='output6' or node.name=='output7' or node.name=='output8' or node.name=='output9':

                self.errors[node.name] = self.errors[input_signal.name][0][node.index]


            else:
                self.errors[node.name] = self.errors[input_signal.name][node.index]

    def visit_Concat(self, node: Concat):

        Visitor.generic_visit(self, node)

        # print("CONCAT", node)
        self.handleIB(node)
        inputs = self.getChildren(node)
        # print(inputs)
        #


        if self.train_MNIST:
            precisions = []

            counter = 0


            if isinstance(inputs,list):
                for i in inputs:

                    precisions.append(copy.deepcopy(self.errors[i.name]))
                    self.errors[node.name] = precisions
                # print(len(self.errors[node.name]))
            else:
                precisions = []
                precisions.append(copy.deepcopy(self.errors[inputs.name]))
                self.errors[node.name] = precisions




    def visit_Reduce(self, node: Reduce):

        Visitor.generic_visit(self, node)
        # print("REDUCE", node)
        self.handleIB(node)
        input_vector = self.getChildren(node)

        if self.train_MNIST:
            self.errors[node.name] = PrecisionNode.reduce(
                self.errors[input_vector.name], node.name)

    def visit_LookupTable(self, node: LookupTable):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        input_signal = self.getChildren(node)
        node.child = input_signal

        if self.calculate_IB:
            self.area_fn += f"+1 * (2 ** ({self.IBs[input_signal.name]} + {input_signal.name})) * ({node.name} + {self.IBs[node.name]})"
            self.errors[node.name] = PrecisionNode(
                self.errors[input_signal.name].val, node.name, self.errors[input_signal.name].error)
        else:
            self.area_fn += f"+1 * (2 ** ({input_signal.name} + {input_signal.name})) * ({node.name} + {node.name})"

        # if self.calculate_IB:
        #     self.area_fn += f"+1 * ({node.numel}) * ({node.name} + {self.IBs[node.name]})"

        # else:
        #     self.area_fn += f"+1 * ({node.numel}) * ({node.name} + {node.name}_ib)"


class BitFlowOptimizer():

    def __init__(self, evaluator, outputs):



        node_values = evaluator.node_values

        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        self.intervals = visitor.intervals
        self.visitor = visitor
        self.error_fn = ""
        self.ufb_fn = ""
        self.optim_error_fn = " >= "

        self.error_fns = []
        self.ufb_fns = []
        self.optim_error_fns = []

        for output in outputs:
            # print(output)
            for i in range(1):
                for j in range(10):


                    self.error_fn += f"+2**(-{outputs[output]}-1) - (" + \
                                     visitor.errors[output].getExecutableError() + ")"
                    self.optim_error_fn = f"+ 2**(-{outputs[output]}-1)" + \
                                          self.optim_error_fn + \
                                          visitor.errors[output].getExecutableError()
                    self.ufb_fn += visitor.errors[output].getExecutableUFB()

                    # self.error_fn += f"+2**(-{outputs[output][i][j]}-1) - (" + \
                    #                  visitor.errors[output][i][j].getExecutableError() + ")"
                    # self.optim_error_fn = f"+ 2**(-{outputs[output][i][j]}-1)" + \
                    #                       self.optim_error_fn + \
                    #                       visitor.errors[output][i][j].getExecutableError()
                    # self.ufb_fn += visitor.errors[output][i][j].getExecutableUFB()

            #self.error_fn = '+2**(-1.0-1) - (+1.0*2**(-X_input_9_0-1)+1.0*2**(-weight_input_0_0-1)+1*(+1*2**(-weight_input_0_0-1))*(+1*2**(-X_input_9_0-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_1-1)+1.0*2**(-weight_input_0_1-1)+1*(+1*2**(-weight_input_0_1-1))*(+1*2**(-X_input_9_1-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_2-1)+1.0*2**(-weight_input_0_2-1)+1*(+1*2**(-weight_input_0_2-1))*(+1*2**(-X_input_9_2-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_3-1)+1.0*2**(-weight_input_0_3-1)+1*(+1*2**(-weight_input_0_3-1))*(+1*2**(-X_input_9_3-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_4-1)+1.0*2**(-weight_input_0_4-1)+1*(+1*2**(-weight_input_0_4-1))*(+1*2**(-X_input_9_4-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_5-1)+1.0*2**(-weight_input_0_5-1)+1*(+1*2**(-weight_input_0_5-1))*(+1*2**(-X_input_9_5-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_6-1)+1.0*2**(-weight_input_0_6-1)+1*(+1*2**(-weight_input_0_6-1))*(+1*2**(-X_input_9_6-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_7-1)+1.0*2**(-weight_input_0_7-1)+1*(+1*2**(-weight_input_0_7-1))*(+1*2**(-X_input_9_7-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_8-1)+1.0*2**(-weight_input_0_8-1)+1*(+1*2**(-weight_input_0_8-1))*(+1*2**(-X_input_9_8-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1)+1.0*2**(-X_input_9_9-1)+1.0*2**(-weight_input_0_9-1)+1*(+1*2**(-weight_input_0_9-1))*(+1*2**(-X_input_9_9-1))+1*2**(-X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0)-1))'

            self.error_fns.append(f"+2**(-{outputs[output]}-1) - (" +
                                  visitor.errors[output].getExecutableError() + ")")
            self.optim_error_fns.append(f"+ 2**(-{outputs[output]}-1) >= " +
                                        visitor.errors[output].getExecutableError())
            self.ufb_fns.append(visitor.errors[output].getExecutableUFB())

            #self.error_fn = self.error_fn[1:]
        self.area_fn = visitor.area_fn[1:]
            # print(len(visitor.area_fn[0]))
        self.outputs = outputs

        # print(len(self.error_fns))
        # print(self.error_fns)
        print(f"ERROR EQ: {self.error_fns}")
        print(f"AREA EQ: {self.area_fn}")


        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        self.vars = vars

    def calculateInitialValues(self):
        # print("CALCULATING INITIAL VALUES USING UFB METHOD...")
        # bnd = f"{-2**(-self.output_precision-1)} == 0"
        bnd = ""
        for output in self.outputs:

            for i in range(1):
                for j in range(10):
                    bnd += f"{-2 ** (-self.outputs[output] - 1)}"

                    # bnd += f"{-2 ** (-self.outputs[output][i][j] - 1)}"
        self.ufb_fn += bnd

        print("*********")
        print(self.ufb_fn)
        print("*********")
        # assert 0



        exec(f'''def UFBOptimizerFn(UFB):
             return  {self.ufb_fn}''', globals())
        print('EXITED UFB')

        sol = ceil(fsolve(UFBOptimizerFn, 0.01))
        self.initial = sol

    def solve(self):

        self.calculateInitialValues()
        print("SOLVING AREA/ERROR...")
        self.error_fn = f"2**(-{self.output_precision}-1)>=" + self.error_fn

        print(f"ERROR EQ: {self.optim_error_fn}")
        print(f"AREA EQ: {self.area_fn}")
        print(f"-----------")

        filtered_vars = []
        for var in self.vars:
            if var not in self.outputs:
                filtered_vars.append(var)

        exec(f'''def ErrorConstraintFn(x):
             {','.join(filtered_vars)} = x
             return  {self.error_fns}''', globals())

        exec(f'''def AreaOptimizerFn(x):
             {','.join(filtered_vars)} = x
             return  {self.area_fn}''', globals())

        x0 = [self.initial for i in range(len(filtered_vars))]
        bounds = [(0, 64) for i in range(len(filtered_vars))]

        con = {'type': 'ineq', 'fun': ErrorConstraintFn}

        # note: minimize uses SLSQP by default but I specify it to be explicit; we're using basinhopping to find the global minimum while using SLSQP to find local minima
        minimizer_kwargs = {'constraints': (
            [con]), 'bounds': bounds, 'method': "SLSQP"}
        solution = basinhopping(AreaOptimizerFn, x0,
                                minimizer_kwargs=minimizer_kwargs)

        sols = dict(zip(filtered_vars, solution.x))

        for key in sols:
            sols[key] = ceil(sols[key])
            print(f"{key}: {sols[key]}")

        self.fb_sols = sols
