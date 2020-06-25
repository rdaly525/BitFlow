from Precision import PrecisionNode, FPEpsilon, FPEpsilonMultiplier
from copy import deepcopy
from sympy import *
from sympy.abc import _clash1

class PrecisionOptimizer():
    def __init__(self, precision_node):
        assert(isinstance(precision_node, PrecisionNode))
        self.node = precision_node

    def maxError(self, error):
        maxErr = deepcopy(error)
        for err in maxErr:
            err.val = abs(err.val)
            if isinstance(err, FPEpsilonMultiplier):
                err.Ex = self.maxError(err.Ex)
                err.Ey = self.maxError(err.Ey)
        return maxErr

    def constructErrorFn(self, errors):
        myfn = ""
        for err in errors:
            if isinstance(err, FPEpsilon):
                myfn += f"+{err.val}*2**(-{err.node}-1)" if err.val > 0 else f"{err.val}*2**(-{err.node}-1)"
            elif isinstance(err, FPEpsilonMultiplier):
                myfn += f"+{err.val}*("
                myfn += self.constructErrorFn(err.Ex)
                myfn += ")*("
                myfn += self.constructErrorFn(err.Ey)
                myfn += ")"
        return myfn

    def solve(self):
        maxErr = self.maxError(self.node.error)
        self.node.error = maxErr
        myfn = f"2 ** (-{self.node.symbol}) >=" +  self.constructErrorFn(self.node.error)
        myfn = sympify(myfn, _clash1)


        print(myfn)

a = PrecisionNode(2, "a", [])
b = PrecisionNode(3, "b", [])
c = PrecisionNode(4, "c", [])

d = a.mul(b, "d")
e = d.add(c, "e")
z = e.sub(b, "z")

po = PrecisionOptimizer(z)
po.solve()
