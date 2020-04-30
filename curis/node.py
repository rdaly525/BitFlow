from .visitor import Visited, Dag
import abc
import typing as tp

#Passes will be run on this
class DagNode(Visited):
    def __init__(self, name, *children) :
        self.name = name
        self.set_children(*children)

    def set_children(self, *children):
        self._children = children

    def children(self):
        yield from self._children

    def inputs(self):
        return self._children

    def __str__(self):
        return self.name

    def copy(self):
        return type(self)(*self.children(), name=self.name)

    def __add__(self, rhs):
        assert isinstance(rhs, DagNode)
        return Add(self, rhs)

    def __sub__(self, rhs):
        assert isinstance(rhs, DagNode)
        return Sub(self, rhs)

    def __mul__(self, rhs):
        assert isinstance(rhs, DagNode)
        return Mul(self, rhs)


class Input(DagNode):
    def __init__(self, name):
        super().__init__(name)

    def eval(self, input_values):
        if self in input_values:
            return input_values[self]
        raise ValueError("Missing Input")

class Constant(DagNode):
    def __init__(self, value, name=None):
        self.value = value
        if name is None:
            name = str(value)
        super().__init__(name)

    def eval(self, _):
        return self.value


class Add(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_add_{b.name}"
        super().__init__(name, a, b)

    def eval(self, input_values):
        a, b = tuple(self.children())
        a_val = a.eval(input_values)
        b_val = b.eval(input_values)
        return a_val + b_val

class Sub(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_sub_{b.name}"
        super().__init__(name, a, b)

    def eval(self, input_values):
        a, b = tuple(self.children())
        a_val = a.eval(input_values)
        b_val = b.eval(input_values)
        return a_val - b_val

class Mul(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_mul_{b.name}"
        super().__init__(name, a, b)

    def eval(self, input_values):
        a, b = tuple(self.children())
        a_val = a.eval(input_values)
        b_val = b.eval(input_values)
        return a_val + b_val

class Dag:
    def __init__(self, output: DagNode, inputs: tp.List[DagNode]):
        self.inputs = inputs
        self._parents = [output]

    def parents(self):
        yield from self._parents

    @property
    def num_outputs(self):
        return self.num_parents

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def num_parents(self):
        return len(self._parents)


    def eval(self, *input_values):
        assert len(input_values) == len(self.inputs)
        input_dict = {self.inputs[i]:input_values[i] for i in range(len(input_values))}
        return self._parents[0].eval(input_dict);
