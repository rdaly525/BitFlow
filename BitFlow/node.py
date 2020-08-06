from DagVisitor import Visited, AbstractDag
import abc
import typing as tp


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

    # def __dotproduct__(self, rhs):
    #     assert isinstance(rhs, DagNode)
    #     return DotProduct(self, rhs)

    def __concat__(self, rhs):
        return Concat(self, rhs)

    def __getitem__(self, rhs):
        #assert isinstance(rhs,int)
        return Select(self, rhs)

    def __len__(self):
        return Len(self)

    def __reduce__(self):
        return Reduce(self)

class Input(DagNode):
    def __init__(self, name):
        super().__init__(name)

class Output(DagNode):
    def __init__(self, name):
        super().__init__(name)

class Len(DagNode):
    def __init__(self, a: DagNode, name=None):
        if name is None:
            name = f"{a.name}_len"
        super().__init__(name, a)

class Constant(DagNode):
    def __init__(self, value, name=None):
        self.value = value
        if name is None:
            name = str(value)
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

# class DotProduct(DagNode):
#     def __init__(self, a: DagNode, b: DagNode, concat_dim, name=None):
#         self.concat_dim = concat_dim
#         if name is None:
#             name = f"{a.name}_dotproduct_{b.name}"
#         super().__init__(name, a, b)

class Concat(DagNode):
    def __init__(self, *args, concat_dim, name=None):
        self.concat_dim = concat_dim
        if name is None:
            name = f"_concat"
        super().__init__(name, *args)


class Select(DagNode):
    def __init__(self, a: DagNode, index, name=None):
        self.index=index

        if name is None:
            name = f"{a.name}_getitem_{str(index)}"
        super().__init__(name, a)

class Round(DagNode):
    def __init__(self, val: DagNode, prec: DagNode, name=None):
        if name is None:
            name = f"{val.name}_round_{prec.name}"
        super().__init__(name, val, prec)

class Reduce(DagNode):
    def __init__(self, a: DagNode, reduce_dim, size,  name=None):
        self.reduce_dim = reduce_dim
        self.size = size
        if name is None:
            name = f"{a.name}_reduce_{size.name}"
        super().__init__(name, a)


class Relu(DagNode):
    def __init__(self, a: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_relu"
        super().__init__(name, a)


class Tanh(DagNode):
    def __init__(self, a: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_tanh"
        super().__init__(name, a)



class Dag(AbstractDag):
    def __init__(self, outputs: tp.List[DagNode], inputs: tp.List[DagNode]):
        assert isinstance(outputs, list)
        assert isinstance(inputs, list)
        self.inputs = inputs
        self.outputs= outputs
        super().__init__(*outputs)
