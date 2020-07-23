from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode, Select
from DagVisitor import Visitor
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil
from .Precision import PrecisionNode
from scipy.optimize import fsolve, minimize, basinhopping


class BitFlowVisitor(Visitor):
    def __init__(self, node_values, calculate_IB=True):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}
        self.area_fn = ""
        self.calculate_IB = calculate_IB

    def handleIB(self, node):
        if self.calculate_IB:
            ib = 0
            x = self.node_values[node]
            if isinstance(x, Interval):
                alpha = 2 if (log2(abs(x.hi)).is_integer()) else 1
                ib = ceil(log2(max(abs(x.lo), abs(x.hi)))) + alpha
            else:
                alpha = 2 if (log2(abs(x)).is_integer()) else 1
                ib = ceil(log2(abs(x))) + alpha
            self.IBs[node.name] = int(ib)

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node)

        return (children[0], children[1])

    def visit_Input(self, node: Input):
        self.handleIB(node)

        if self.calculate_IB:
            val = 0
            if isinstance(self.node_values[node], Interval):
                x = self.node_values[node]
                val = max(abs(x.lo), abs(x.hi))
            else:
                val = self.node_values[node]

            self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Select(self, node: Select):
        Visitor.generic_visit(self, node)

    def visit_Constant(self, node: Constant):
        self.handleIB(node)

        if self.calculate_IB:
            val = self.node_values[node]
            self.errors[node.name] = PrecisionNode(val, node.name, [])

    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)

        self.handleIB(node)

        lhs, rhs = self.getChildren(node)

        if self.calculate_IB:
            self.errors[node.name] = self.errors[lhs.name].add(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+max({lhs.name}_ib + {lhs.name}, {rhs.name}_ib + {rhs.name})"

    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB:
            self.errors[node.name] = self.errors[lhs.name].sub(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+max({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+max({lhs.name}_ib + {lhs.name}, {rhs.name}_ib + {rhs.name})"

    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)

        if self.calculate_IB:
            self.errors[node.name] = self.errors[lhs.name].mul(
                self.errors[rhs.name], node.name)

        if self.calculate_IB:
            self.area_fn += f"+1 * ({self.IBs[lhs.name]} + {lhs.name})*({self.IBs[rhs.name]} + {rhs.name})"
        else:
            self.area_fn += f"+1 * ({lhs.name}_ib + {lhs.name})*({rhs.name}_ib + {rhs.name})"


class BitFlowOptimizer():
    def __init__(self, evaluator, outputs):

        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        self.visitor = visitor
        self.error_fn = ""
        self.ufb_fn = ""
        for output in outputs:
            self.error_fn += f"+2**(-{outputs[output]}-1) - (" + \
                visitor.errors[output].getExecutableError() + ")"
            self.ufb_fn += visitor.errors[output].getExecutableUFB()
        self.area_fn = visitor.area_fn[1:]
        self.outputs = outputs

        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        self.vars = vars

    def calculateInitialValues(self):
        #print("CALCULATING INITIAL VALUES USING UFB METHOD...")
        # bnd = f"{-2**(-self.output_precision-1)} == 0"
        bnd = ""
        for output in self.outputs:
            bnd += f"{-2**(-self.outputs[output]-1)}"
        self.ufb_fn += bnd
        #print(f"UFB EQ: {self.ufb_fn}")
        # print(f"-----------")

        exec(f'''def UFBOptimizerFn(UFB):
             return  {self.ufb_fn}''', globals())

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
        self.calculateInitialValues()
        print("SOLVING AREA/ERROR...")
        # self.error_fn = f"2**(-{self.output_precision}-1)>=" + self.error_fn

        print(f"ERROR EQ: {self.error_fn}")
        print(f"AREA EQ: {self.area_fn}")
        print(f"-----------")

        filtered_vars = []
        for var in self.vars:
            if var not in self.outputs:
                filtered_vars.append(var)

        exec(f'''def ErrorConstraintFn(x):
             {','.join(filtered_vars)} = x
             return  {self.error_fn}''', globals())

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

        # namespace = {"m": GEKKO()}
        # m = namespace["m"]
        # m.options.IMODE=2
        # m.options.SOLVER=3
        #
        # filtered_vars = []
        # for var in self.vars:
        #     if var != self.output:
        #         filtered_vars.append(var)
        #
        # vars_init = ','.join(filtered_vars) + f" = [m.Var(value={self.initial}, integer=True, lb=0, ub=64) for i in range({len(filtered_vars)})]"
        # exec(vars_init, namespace)
        #
        # exec(f'''def ErrorOptimizerFn({','.join(filtered_vars)}):
        #     return  {self.error_fn}''', namespace)
        #
        # exec(f'''def AreaOptimizerFn({','.join(filtered_vars)}):
        #     return  {self.area_fn}''', namespace)
        #
        # params = [namespace[v] for v in filtered_vars]
        #
        # m.Equation(namespace["ErrorOptimizerFn"](*params))
        # m.Obj(namespace["AreaOptimizerFn"](*params))
        # m.solve(disp=True)
        #
        # sols = dict(zip(filtered_vars, params))
        #
        # for key in sols:
        #     sols[key] = ceil(sols[key].value[0])
        #     print(f"{key}: {sols[key]}")
        #
        # self.fb_sols = sols
