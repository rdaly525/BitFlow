from .node import Input, Constant, Dag, Add, Sub, Mul, DagNode
from DagVisitor import Visitor
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from math import log2, ceil
from .Precision import PrecisionNode
from gekko import GEKKO


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

class BitFlowVisitor(Visitor):
    def __init__(self, node_values):
        self.node_values = node_values
        self.errors = {}
        self.IBs = {}
        self.area_fn = ""

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

    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].add(self.errors[rhs.name], node.name)
        self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"


    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].sub(self.errors[rhs.name], node.name)
        self.area_fn += f"+m.max2({self.IBs[lhs.name]} + {lhs.name}, {self.IBs[rhs.name]} + {rhs.name})"


    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)

        self.handleIB(node)
        lhs, rhs = self.getChildren(node)
        self.errors[node.name] = self.errors[lhs.name].mul(self.errors[rhs.name], node.name)
        self.area_fn += f"+({self.IBs[lhs.name]} + {lhs.name})*({self.IBs[rhs.name]} + {rhs.name})"

class BitFlowOptimizer():
    def __init__(self, evaluator, output, output_precision):

        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        self.visitor = visitor
        self.error_fn = visitor.errors[output].getExecutableError()
        self.ufb_fn = visitor.errors[output].getExecutableUFB()
        self.area_fn = visitor.area_fn[1:]
        self.output_precision = output_precision
        self.output = output

        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        self.vars = vars


    def calculateInitialValues(self):
        print("CALCULATING INITIAL VALUES USING UFB METHOD...")
        bnd = f"{-2**(-self.output_precision-1)} == 0"
        self.ufb_fn += bnd
        print(f"UFBB EQ: {self.ufb_fn}")
        print(f"-----------")

        m = GEKKO()
        UFB = m.Var(value=0,integer=True)

        exec(f'''def UFBOptimizerFn(UFB):
            return  {self.ufb_fn}''', globals())

        m.Equation(UFBOptimizerFn(UFB))
        m.solve(disp=False)

        sol = ceil(UFB.value[0])
        self.initial = sol
        print(f"UFB = {sol}\n")


    def solve(self):
        self.calculateInitialValues()
        print("SOLVING AREA/ERROR...")
        self.error_fn = f"2**(-{self.output_precision}-1)>=" + self.error_fn
        print(f"ERROR EQ: {self.error_fn}")
        print(f"AREA EQ: {self.area_fn}")
        print(f"-----------")

        namespace = {"m": GEKKO()}
        m = namespace["m"]

        filtered_vars = []
        for var in self.vars:
            if var != self.output:
                filtered_vars.append(var)

        vars_init = ','.join(filtered_vars) + f" = [m.Var(value={self.initial}, lb=0, ub=64) for i in range({len(filtered_vars)})]"
        exec(vars_init, namespace)

        exec(f'''def ErrorOptimizerFn({','.join(filtered_vars)}):
            return  {self.error_fn}''', namespace)

        exec(f'''def AreaOptimizerFn({','.join(filtered_vars)}):
            return  {self.area_fn}''', namespace)

        params = [namespace[v] for v in filtered_vars]

        m.Equation(namespace["ErrorOptimizerFn"](*params))
        m.Obj(namespace["AreaOptimizerFn"](*params))
        m.solve(disp=False)

        sols = dict(zip(filtered_vars, params))

        for key in sols:
            sols[key] = ceil(sols[key].value[0])
            print(f"{key}: {sols[key]}")


def test_print():
    fig3 = gen_fig3()
    evaluator = IAEval(fig3)

    a, b = Interval(-3, 2), Interval(4, 8)
    evaluator.eval(a=a, b=b)

    bfo = BitFlowOptimizer(evaluator, 'z', 8)
    bfo.solve()

test_print()