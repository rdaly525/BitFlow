from DagVisitor import Visited, AbstractDag
from BitFlow.utils import LUTGenerator
import abc
import typing as tp
import random


# Passes will be run on this
class DagNode(Visited):
    def __init__(self, name, *children):
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

    def __getitem__(self, rhs):
        assert isinstance(rhs, int)
        return Select(self, rhs)


class Input(DagNode):
    def __init__(self, name):
        super().__init__(name)


class Output(DagNode):
    def __init__(self, name):
        super().__init__(name)


class Constant(DagNode):
    def __init__(self, value, name=None):
        self.value = value
        if name is None:
            name = f"Constant_{random.randint(0, 100000)}"
        super().__init__(name)


class Add(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_add_{b.name}"
        super().__init__(name, a, b)


class Sub(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_sub_{b.name}"
        super().__init__(name, a, b)


class Mul(DagNode):
    def __init__(self, a: DagNode, b: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_mul_{b.name}"
        super().__init__(name, a, b)


class Select(DagNode):
    def __init__(self, a: DagNode, index, name=None):
        self.index = index
        if name is None:
            name = f"{a.name}_getitem_{str(index)}"
        super().__init__(name, a)


class Round(DagNode):
    def __init__(self, val: DagNode, prec: DagNode, rng: DagNode, name=None):
        self.prec = prec
        self.rng = rng
        if name is None:
            name = f"{val.name}_round_{prec.name}_{rng}"
        super().__init__(name, val, prec, rng)


class LookupTable(DagNode):
    def __init__(self, func, a: DagNode, name=None):
        self.func = func
        if name is None:
            name = f"lookup_{a.name}_{func.__name__}"
        super().__init__(name, a)


class Dag(AbstractDag):
    def __init__(self, outputs: tp.List[DagNode], inputs: tp.List[DagNode]):
        assert isinstance(outputs, list)
        assert isinstance(inputs, list)
        self.inputs = inputs
        self.outputs = outputs
        super().__init__(*outputs)
