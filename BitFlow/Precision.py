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
        return f" +{self.val} * 2^(-FB{self.node} - 1) ε_{self.node}" if self.val >= 0 else f" {self.val} * 2^(FB{self.node} - 1) ε_{self.node}"


class PrecisionNode:
    def __init__(self, val, symbol, error):
        assert isinstance(val, int)
        assert isinstance(symbol, str)
        assert isinstance(error, list)

        self.val = val
        self.symbol = symbol
        self.error = error
        self.error.append(FPEpsilon(symbol))
        self.output = ""

    def generate_print(self, errors):
        for err in errors:
            if (isinstance(err, tuple)):
                self.output += " ("
                self.generate_print(err[0])
                self.output += ")("
                self.generate_print(err[1])
                self.output += ") "
            else:
                self.output += str(err)
        return self.output

    def __str__(self):
        print(f"NODE {self.symbol}:")
        self.output = f"{self.val}"
        return self.generate_print(self.error) + "\n"

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

        mixed_err = (copy.deepcopy(rhs.error), copy.deepcopy(self.error))
        print(mixed_err[0][0])

        rhs_error = copy.deepcopy(rhs.error)
        for (i, error) in enumerate(rhs_error):
            rhs_error[i].val *= self.val

        lhs_error = copy.deepcopy(self.error)
        for (i, error) in enumerate(lhs_error):
            lhs_error[i].val *= rhs.val

        total_err = lhs_error + rhs_error
        total_err.append(mixed_err)

        # Store multiplication of errors as a tuple whose first element is E_x and second element is E_y
        return PrecisionNode(self.val * rhs.val, symbol, total_err)
