from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Select, LookupTable, BitShift, Concat, Reduce
from DagVisitor import Visitor
from .IA import Interval
from .AA import AInterval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil, floor
from .Precision import PrecisionNode
from scipy.optimize import fsolve, minimize, basinhopping
from gekko import GEKKO
import torch
import copy


class BitFlowVisitor(Visitor):
    def __init__(self, node_values, calculate_IB=True):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}
        self.intervals = {}
        self.area_fn = ""
        self.ep_id = 0

        self.train_MNIST = False
        if self.train_MNIST:
            self.calculate_IB = False
        else:
            self.calculate_IB = calculate_IB

    def handleIB(self, node):
        if self.calculate_IB:
            x = self.node_values[node]
            # print(f"{node}: {x}")
            if isinstance(x, Interval):
                alpha = 2 if (log2(abs(x.hi)).is_integer()) else 1
                ib = ceil(log2(max(abs(x.lo), abs(x.hi)))) + alpha
                self.IBs[node.name] = int(ib)
            elif isinstance(x, AInterval):
                interval = x.to_interval()
                ib = log2(max(abs(interval.lo), abs(interval.hi)))
                alpha = 2 if ib.is_integer() else 1
                if max(abs(interval.lo), abs(interval.hi)) < 1:
                    ib = 1
                    alpha = 0
                print(f"{node.name}: {interval}, {int(ceil(ib) + alpha)}")
                self.IBs[node.name] = int(ceil(ib) + alpha)
                self.intervals[node.name + "_round"] = interval

            else:
                ib = abs(x)
                if ib < 1:
                    self.IBs[node.name] = 1
                    return
                alpha = 2 if ib >= 1. and log2(ib).is_integer() else 1
                self.IBs[node.name] = int(ceil(log2(ib)) + alpha)
                self.intervals[node.name +
                               "_round"] = Interval(-abs(x), abs(x))
                # if (x == 0.):
                #     self.IBs[node.name] = 1
                # elif (x < 1.):
                #     self.IBs[node.name] = 0
                # elif isinstance(x, list):
                #     for (ind, val) in enumerate(x):
                #         alpha = 2 if (log2(abs(val)).is_integer()) else 1
                #         ib = ceil(log2(abs(val))) + alpha
                #         self.IBs[f"{node.name}_getitem_{ind}"] = ib
                #     return
                # else:
                #     alpha = 2 if (log2(abs(x)).is_integer()) else 1
                #     ib = ceil(log2(abs(x))) + alpha
                #     self.IBs[node.name] = int(ib)

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)
        if len(children) == 1:
            return children[0]
        return children

    def visit_Input(self, node: Input):
        self.handleIB(node)

        if self.train_MNIST:
            error_mat = []
            for row in range(28):
                error_mat.append([])
                for col in range(28):
                    error_mat[row].append(PrecisionNode(
                        1., f"{node.name}_input_{row}_{col}", []))
            print(node.name)
            self.errors[node.name] = error_mat
            return

        if self.calculate_IB:
            val = 0
            if isinstance(self.node_values[node], Interval):
                x = self.node_values[node]
                val = max(abs(x.lo), abs(x.hi))
            elif isinstance(self.node_values[node], AInterval):
                x = self.node_values[node].to_interval()
                val = max(abs(x.lo), abs(x.hi))
            else:
                val = self.node_values[node]

            self.errors[node.name] = PrecisionNode(abs(val), node.name, [])

    def visit_Constant(self, node: Constant):
        self.handleIB(node)

        if self.calculate_IB or self.train_MNIST:
            val = self.node_values[node]
            self.errors[node.name] = PrecisionNode(abs(val), node.name, [])

    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)

        self.handleIB(node)

        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:
            self.errors[node.name] = self.errors[lhs.name].add(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+1 * max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * max({lhs.name}_ib + {lhs.name}, {rhs.name}_ib + {rhs.name})"

    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:
            self.errors[node.name] = self.errors[lhs.name].sub(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+1 * max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * max({lhs.name}_ib + {lhs.name}, {rhs.name}_ib + {rhs.name})"

    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB or self.train_MNIST:
            self.errors[node.name] = self.errors[lhs.name].mul(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+1 * ({self.IBs[lhs.name]} + {lhs.name})*({self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * ({lhs.name}_ib + {lhs.name})*({rhs.name}_ib + {rhs.name})"

    def visit_BitShift(self, node: BitShift):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB:
            self.errors[node.name] = self.errors[lhs.name].mul(
                self.errors[rhs.name], node.name)

    def visit_Select(self, node: Select):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        input_signal = self.getChildren(node)

        if self.train_MNIST:
            self.errors[node.name] = self.errors[input_signal.name][node.index]

    def visit_Concat(self, node: Concat):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        inputs = self.getChildren(node)

        if self.train_MNIST:
            precisions = []
            for i in inputs:
                precisions.append(copy.deepcopy(self.errors[i.name]))
            self.errors[node.name] = precisions

    def visit_Reduce(self, node: Reduce):
        Visitor.generic_visit(self, node)

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
            if node.precision >= 0.:
                self.area_fn += f"+1 * (2 ** ({self.IBs[input_signal.name]} + {input_signal.name})) * ({node.precision} + {self.IBs[node.name]})"
            else:
                self.area_fn += f"+1 * (2 ** ({self.IBs[input_signal.name]} + {input_signal.name})) * ({node.name} + {self.IBs[node.name]})"
            self.errors[node.name] = PrecisionNode(
                self.errors[input_signal.name].val, node.name, self.errors[input_signal.name].error)
        else:
            if node.precision >= 0.:
                # TODO: the 5. here shouldn't be hardcoded!
                self.area_fn += f"+1 * (2 ** ({input_signal.name} + {input_signal.name}_ib)) * ({node.precision} + 5.)"
            else:
                self.area_fn += f"+1 * (2 ** ({input_signal.name} + {input_signal.name}_ib)) * ({node.name} + {node.name}_ib)"

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
        self.error_fns = []
        self.ufb_fns = []
        self.optim_error_fns = []
        for output in outputs:
            self.error_fns.append(f"+2**(-{outputs[output]}-1) - (" +
                                  visitor.errors[output].getExecutableError() + ")")
            self.optim_error_fns.append(f"+ 2**(-{outputs[output]}-1) >= " +
                                        visitor.errors[output].getExecutableError())
            self.ufb_fns.append(visitor.errors[output].getExecutableUFB())
        self.area_fn = visitor.area_fn[1:]
        self.outputs = outputs

        # print(f"ERROR EQ: {self.error_fns}")
        # print(f"AREA EQ: {self.area_fn}")

        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        self.vars = vars

    def calculateInitialValues(self):
        # print("CALCULATING INITIAL VALUES USING UFB METHOD...")
        # bnd = f"{-2**(-self.output_precision-1)} == 0"
        for (ind, output) in enumerate(self.outputs):
            self.ufb_fns[ind] += f"{-2**(-self.outputs[output]-1)}"

        #print(f"UFB EQ: {self.ufb_fns}")
        # print(f"-----------")
        exec(f'''def UFBOptimizerFn(UFB):
             return  [{','.join(self.ufb_fns)}] if {len(self.ufb_fns)} == 0 else {self.ufb_fns[0]}''', globals())

        sol = ceil(fsolve(UFBOptimizerFn, 0.01))
        self.initial = sol

        # m = GEKKO()
        # UFB = m.Var(value=0,integer=True)
        # m.options.IMODE=2
        # m.options.SOLVER=3
        #
        # exec(f'''def UFBOptimizerFn(UFB):
        #     return  {self.ufb_fn}''', globals())
        #
        # m.Equation(UFBOptimizerFn(UFB))
        # m.solve(disp=True)
        #
        # sol = ceil(UFB.value[0])
        # self.initial = sol
        # print(f"UFB = {sol}\n")

    def solve(self):
        # self.calculateInitialValues()
        print("SOLVING AREA/ERROR...")
        # self.error_fn = f"2**(-{self.output_precision}-1)>=" + self.error_fn

        print(f"ERROR EQ: {self.optim_error_fns}")
        print(f"AREA EQ: {self.area_fn}")
        print(f"-----------")

        # filtered_vars = []
        # for var in self.vars:
        #     if var not in self.outputs:
        #         filtered_vars.append(var)

        # functions = {}

        # exec(f'''def ErrorConstraintFn(x):
        #      {','.join(filtered_vars)} = x
        #      return  [{','.join(self.error_fns)}] if {len(self.error_fns)} == 0 else {self.error_fns[0]}''', functions)

        # exec(f'''def AreaOptimizerFn(x):
        #      {','.join(filtered_vars)} = x
        #      return  {self.area_fn}''', functions)

        # x0 = [self.initial for i in range(len(filtered_vars))]
        # bounds = [(0, 64) for i in range(len(filtered_vars))]

        # cons = []
        # for ind, fn in enumerate(self.error_fns):
        #     exec(f'''def ErrorConstraintFn_{ind}(x):
        #      {','.join(filtered_vars)} = x
        #      return  {fn}''', functions)
        #     con = {'type': 'ineq',
        #            'fun': functions[f"ErrorConstraintFn_{ind}"]}
        #     cons.append(con)

        # # note: minimize uses SLSQP by default but I specify it to be explicit; we're using basinhopping to find the global minimum while using SLSQP to find local minima
        # minimizer_kwargs = {'constraints': (
        #     cons), 'bounds': bounds, 'method': "SLSQP"}
        # solution = basinhopping(functions[f"AreaOptimizerFn"], x0,
        #                         minimizer_kwargs=minimizer_kwargs)

        # sols = dict(zip(filtered_vars, solution.x))

        # for key in sols:
        #     sols[key] = ceil(sols[key])
        #     print(f"{key}: {sols[key]}")

        # self.fb_sols = sols

        # test = list(self.fb_sols.values())

        # # err = functions["ErrorConstraintFn_0"]
        # # err1 = functions["ErrorConstraintFn_1"]
        # # err2 = functions["ErrorConstraintFn_2"]
        # area = functions["AreaOptimizerFn"]

        # # print(f"ERROR: {err(test)}")
        # # print(f"ERROR: {err1(test)}")
        # # print(f"ERROR: {err2(test)}")
        # print(f"AREA: {area(test)}")

        namespace = {"m": GEKKO()}
        m = namespace["m"]
        m.options.IMODE = 2
        m.options.SOLVER = 3

        filtered_vars = []
        for var in self.vars:
            if var not in self.outputs:
                filtered_vars.append(var)

        vars_init = ','.join(
            filtered_vars) + f" = [m.Var(value={self.initial}, integer=True, lb=0, ub=64) for i in range({len(filtered_vars)})]"
        exec(vars_init, namespace)

        exec(f'''def ErrorOptimizerFn({','.join(filtered_vars)}):
            return  {self.optim_error_fns}''', namespace)

        exec(f'''def AreaOptimizerFn({','.join(filtered_vars)}):
            return  {self.area_fn.replace("max", "m.max2")}''', namespace)

        exec(f'''def AreaOptimizerFn0({','.join(filtered_vars)}):
            return  {self.area_fn}''', namespace)

        params = [namespace[v] for v in filtered_vars]

        for ind, fn in enumerate(self.optim_error_fns):
            exec(f'''def ErrorConstraintFn_{ind}({','.join(filtered_vars)}):
                    return  {fn}''', namespace)
            m.Equation(namespace[f"ErrorConstraintFn_{ind}"](*params))

        m.Obj(namespace["AreaOptimizerFn"](*params))
        m.solve(disp=True)

        sols = dict(zip(filtered_vars, params))

        for key in sols:
            sols[key] = ceil(sols[key].value[0])
            print(f"{key}: {sols[key]}")

        self.fb_sols = sols

        test = list(self.fb_sols.values())

        #err = namespace["ErrorConstraintFn_0"]
        # err1 = namespace["ErrorConstraintFn_1"]
        # err2 = namespace["ErrorConstraintFn_2"]
        area = namespace["AreaOptimizerFn0"]

        # print(f"ERROR: {err(*test)}")
        # print(f"ERROR: {err1(*test)}")
        # print(f"ERROR: {err2(*test)}")
        print(f"AREA: {area(*test)}")
