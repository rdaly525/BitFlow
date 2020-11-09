from .node import Input, Output, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, LookupTable
import numpy as np

from DagVisitor import Visitor


class BitFlow2Verilog(Visitor):
    def __init__(self, moduleName, P, R, order, unroundedDAG, outputs):

        bitwidths = list(zip(P, R))
        self.graph = dict(zip(order, bitwidths))
        self.outputNames = list(outputs.keys())
        self.outputs = []

        self.inputs = []
        self.constants = []
        self.in_decl = []
        self.out_decl = []
        self.ordered_vlog = []
        self.unroundedDAG = unroundedDAG
        self.name = moduleName

        for output in outputs:
            print(output)
            # FIX RANGE BITS FOR OUTPUTS
            self.graph[output] = (outputs[output], 8.)
            # self.outputs.append(f"output [{int(outputs[output]) + 8 - 1}:0] {output}")
            self.outputs.append(f"\n\t`OUTPUT_REAL({output})")
            self.out_decl.append(f"\n\t`DECL_REAL({output})")

    def evaluate(self):
        self.run(self.unroundedDAG)
        print(self.inputs)
        print(self.ordered_vlog)

        code = f"`include \"svreal.sv\"\nmodule {self.name} #({','.join(self.in_decl)}, {', '.join(self.out_decl)}) \n({', '.join(self.inputs)}, {', '.join(self.outputs)}\n); \n \n"
        code += f"{''.join(self.constants)}\n"
        for statement in reversed(self.ordered_vlog):
            if statement[0] != None:
                code += f"\t{statement[0]}"
            code += f"\t{statement[1]}\n"

        code += "\nendmodule"

        file = open(f"{self.name}.v", 'w')
        file.write(code)
        file.close()

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node.name)
        return children

    def calcMinValForBits(self, precision, rng):
        min_val = (-1 * (2 ** (precision.item() + (rng.item() - 1) - 1))
                   * 2 ** -precision.item())
        return min_val

    def codeForOperation(self, node, operation):
        node_vals = self.graph[node.name]
        initialize = None
        if node.name not in self.outputNames:
            initialize = f"`MAKE_GENERIC_REAL({node.name}, {self.calcMinValForBits(node_vals[0], node_vals[1])}, {node_vals[0] + node_vals[1] });\n"

            # initialize = f"wire [{node_vals[0] + node_vals[1] - 1}:0] {node.name};\n"

        # TODO: generalize to combine multiple outs

        if operation == ' + ':
            calculate = f"`ADD_INTO_REAL({self.getChildren(node)[0]}, {self.getChildren(node)[1]}, {node.name});\n"
        elif operation == ' * ':
            calculate = f"`MUL_INTO_REAL({self.getChildren(node)[0]}, {self.getChildren(node)[1]}, {node.name});\n"
        elif operation == ' - ':
            calculate = f"`SUB_INTO_REAL({self.getChildren(node)[0]}, {self.getChildren(node)[1]}, {node.name});\n"
        else:
            assert 0

            # calculate = f"assign {node.name} = {operation.join(self.getChildren(node))};\n"
        self.ordered_vlog.append((initialize, calculate))

    def generic_visit(self, node: DagNode):
        Visitor.generic_visit(self, node)

    def visit_Input(self, node: Input):
        node_vals = self.graph[node.name]
        self.inputs.append(
            f"\n\t`INPUT_REAL({node.name})")
        self.in_decl.append(f"\n\t`DECL_REAL({node.name})")
        # self.inputs.append(
        #     f"input [{node_vals[0] + node_vals[1] - 1}:0] {node.name}")
        Visitor.generic_visit(self, node)

    def visit_Add(self, node: Add):
        self.codeForOperation(node, ' + ')
        Visitor.generic_visit(self, node)

    def visit_Mul(self, node: Mul):
        self.codeForOperation(node, ' * ')
        Visitor.generic_visit(self, node)

    def visit_Constant(self, node: Constant):
        node_vals = self.graph[node.name]
        # self.inputs.append(
        #     f"input [{node_vals[0] + node_vals[1] - 1}:0] {node.name}")
        self.constants.append(
            f"\t`MAKE_CONST_REAL({node.value}, {node.name});\n")
        Visitor.generic_visit(self, node)

    def visit_Sub(self, node: Sub):
        self.codeForOperation(node, ' - ')
        Visitor.generic_visit(self, node)

    # def visit_LookupTable(self, node: LookupTable):
    #     Visitor.generic_visit(self, node)
    #     self.maintainGraph(
    #         node, '#8e44ad', node.func.__name__.replace('np.', "") + "(x)")

    # def visit_Select(self, node: Select):
    #     Visitor.generic_visit(self, node)
    #     self.maintainGraph(node, '#d35400', "W")
