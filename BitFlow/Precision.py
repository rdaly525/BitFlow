import copy

'''
Since every instance of epsilon is only associated with precision bit error term, we can bundle them together under one class.
'''


class FPEpsilon:
    def __init__(self, node, val=1):
        assert isinstance(node, str)
        self.node = node
        self.val = val

    def __str__(self):
        return f" +{self.val}*2^(-FB{self.node}-1)ε_{self.node}" if self.val >= 0 else f" {self.val} * 2^(FB{self.node}-1)ε_{self.node}"

    def __eq__(self, rhs):
        return self.val == rhs.val


'''
The most tricky part is multiplying two errors; this object abstracts those error multipliications
'''


class FPEpsilonMultiplier:
    def __init__(self, Ex, Ey, val=1):
        assert isinstance(Ex, list)
        assert isinstance(Ey, list)
        self.Ex = Ex
        self.Ey = Ey
        self.val = val
        self.output = ""

    def generate_print(self):
        self.output += "("
        for err in self.Ex:
            self.output += str(err)
        self.output += " )("
        for err in self.Ey:
            self.output += str(err)
        self.output += " ) "
        return self.output

    def __str__(self):
        self.output = f" +{self.val}*"
        return self.generate_print()

    def check_error_equality(self, rhs, lhs):
        rhs_error = copy.deepcopy(rhs)
        for err in lhs:
            try:
                i = rhs_error.index(err)
                del rhs_error[i]
            except ValueError:
                return False
        return rhs_error == []

    def __eq__(self, rhs):
        return self.check_error_equality(rhs.Ex, self.Ex) and self.check_error_equality(rhs.Ey, self.Ey) and self.val == rhs.val


class PrecisionNode:
    def __init__(self, val, symbol, error, add_self_error=True):
        assert isinstance(val, (int, float))
        assert isinstance(symbol, str)
        assert isinstance(error, list)

        self.val = val
        self.symbol = symbol
        self.error = error
        if add_self_error:
            self.error.append(FPEpsilon(symbol))
        self.output = ""

    def generate_print(self, errors):
        for err in errors:
            self.output += str(err)
        return self.output

    def __str__(self):
        print(f"NODE {self.symbol}:")
        self.output = f"{self.val}"
        return self.generate_print(self.error) + "\n"

    def constructErrorFn(self, errors):
        myfn = ""
        for err in errors:
            if isinstance(err, FPEpsilon):
                if err.node == self.symbol:
                    return myfn
                myfn += f"+{err.val}*2**(-{err.node}-1)" if err.val >= 0 else f"{err.val}*2**(-{err.node}-1)"
            elif isinstance(err, FPEpsilonMultiplier):
                myfn += f"+{err.val}*("
                myfn += self.constructErrorFn(err.Ex)
                myfn += ")*("
                myfn += self.constructErrorFn(err.Ey)
                myfn += ")"
        return myfn

    def constructUFBFn(self, errors):
        myfn = ""
        for err in errors:
            if isinstance(err, FPEpsilon):
                myfn += f"+{err.val}*2**(-UFB-1)" if err.val >= 0 else f"{err.val}*2**(-UFB-1)"
            elif isinstance(err, FPEpsilonMultiplier):
                myfn += f"+{err.val}*("
                myfn += self.constructUFBFn(err.Ex)
                myfn += ")*("
                myfn += self.constructUFBFn(err.Ey)
                myfn += ")"
        return myfn

    def getExecutableError(self):
        return self.constructErrorFn(self.error)

    def getExecutableUFB(self):
        return self.constructUFBFn(self.error)

    def add(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        return PrecisionNode(self.val + rhs.val, symbol, self.error + rhs.error)

    def sub(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        # negate every element in the rhs error (the one being subtracted)
        subtracted_error = copy.deepcopy(rhs.error)
        for (i, error) in enumerate(subtracted_error):
            subtracted_error[i].val *= -1

        return PrecisionNode(self.val - rhs.val, symbol, self.error + subtracted_error)

    def mul(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        mixed_err = FPEpsilonMultiplier(
            copy.deepcopy(rhs.error), copy.deepcopy(self.error))

        rhs_error = copy.deepcopy(rhs.error)
        for (i, error) in enumerate(rhs_error):
            rhs_error[i].val *= self.val

        lhs_error = copy.deepcopy(self.error)
        for (i, error) in enumerate(lhs_error):
            lhs_error[i].val *= rhs.val

        total_err = lhs_error + rhs_error
        total_err.append(mixed_err)

        return PrecisionNode(self.val * rhs.val, symbol, total_err)

    @staticmethod
    def reduce(list_of_precs, symbol):
        total_err = []
        for prec in list_of_precs:
            total_err += prec.error
        return PrecisionNode(sum([p.val for p in list_of_precs]), symbol, total_err)

    def check_error_equality(self, rhs):
        rhs_error = copy.deepcopy(rhs)
        for err in self.error:
            try:
                i = rhs_error.index(err)
                del rhs_error[i]
            except ValueError:
                return False
        return rhs_error == []

    def __eq__(self, rhs):
        return self.check_error_equality(rhs.error) and self.val == rhs.val