from DagVisitor import Visitor
from abc import abstractmethod
from ..node import Dag, DagNode, Input, Constant, Add, Mul, Select, Output, Round


class AbstractEval(Visitor):
    def __init__(self, dag: Dag):
        self.dag = dag
        self.node_values = {}

    def eval(self, **input_values):
        self.input_values = input_values
        self.node_values = {}
        for dag_input in self.dag.inputs:
            if dag_input.name not in input_values:
                raise ValueError(f"Missing {dag_input} in input values")
        super().run(self.dag)
        outputs = [self.node_values[root] for root in self.dag.roots()]
        #print(outputs, len(outputs), outputs[0])
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def generic_visit(self, node: DagNode):
        Visitor.generic_visit(self, node)
        child_values = [self.node_values[child] for child in node.children()]
        eval_name = f"eval_{node.kind()[0]}"
        assert hasattr(self, eval_name)
        node_val = getattr(self, eval_name)(*child_values, node=node)

        assert node_val is not None
        self.node_values[node] = node_val

    def eval_Input(self, node: DagNode):
        return self.input_values[node.name]

    @abstractmethod
    def eval_Constant(self, node: DagNode):
        pass

    @abstractmethod
    def eval_Add(self, a, b, node: DagNode):
        pass

    @abstractmethod
    def eval_Sub(self, a, b, node: DagNode):
        pass

    @abstractmethod
    def eval_Mul(self, a, b, node: DagNode):
        pass

    @abstractmethod
    def eval_Select(self, a, node: DagNode):
        pass
