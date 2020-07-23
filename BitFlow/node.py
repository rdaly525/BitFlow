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

    def __getitem__(self, rhs):
        assert isinstance(rhs,int)
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
    def __init__(self, a: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_reduce"
        super().__init__(name, a)
    #pass
    #dot product, multiply of two vectors,
    #sum node, summing across top dimension of the array
    #reduce/fold sum
    #dot product will be a library which will stamp out of sum and multiply node
    #reduce is summing along one of the axi

    #assume we have a vector with terms multiplied

    #get number of elements in dagnode?
    #for i number of iterations, add dagnodes



    #def DotProduct(list1, list2):
        #return sum(ElementWiseMultiplication(list1, list2))

    pass

class Relu(DagNode):
    def __init__(self, a: DagNode, *, name=None):
        if name is None:
            name = f"{a.name}_relu"
        super().__init__(name, a)
    #R(z) = max(0,z)


class Tanh(DagNode) :
    #
    # 2/(1+e^-2x) - 1
    #gradient steeper than sigmoid
    #squashes a real-valued number to the range [-1, 1]
    pass



class Dag(AbstractDag):
    def __init__(self, outputs: tp.List[DagNode], inputs: tp.List[DagNode]):
        assert isinstance(outputs, list)
        assert isinstance(inputs, list)
        self.inputs = inputs
        self.outputs= outputs
        super().__init__(*outputs)
