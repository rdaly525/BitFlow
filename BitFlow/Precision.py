from .IA import Interval

'''
Since every instance of epsilon is only associated with precision bit error term, we can bundle them together under one class.
'''
class FPEpsilon:
    def __init__(self, node, value=1):
        assert isinstance(node, str)
        self.node = node
        self.value = value


class PrecisionNode:
    def __init__(self, val, symbol, error=[], *):
        assert isinstance(val, int)
        assert isinstance(symbol, str)
        assert isinstance(error, list)

        self.val = val
        self.symbol = symbol
        self.error = error
        self.error.append(FPEpsilon(symbol))

    def generate_print(errors, mystr):
        for err in errors:
            if (isinstance(err, tuple)):
                mystr += " ("
                generate_print(err[0], mystr)
                mystr += ")("
                generate_print(err[1], mystr)
                mystr += ") "
            else:
                mystr += f" + {err.value} * 2^(FB{err.node} - 1) ε_{err.node}"
        return mystr


    def __str__(self):
        mystr = f"{self.value}"
        return generate_print(self.errors)

    def __add__(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        return PrecisionNode(self.val + rhs.val, symbol, self.error + rhs.error)

    def __sub__(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        subtracted_error = rhs.error.copy()
        for (i, error) in enumerate(subtracted_error):
            subtracted_error[i].value *= -1

        return PrecisionNode(self.val - rhs.val, symbol, self.error + subtracted_error)

    def __mul__(self, rhs, symbol):
        assert isinstance(rhs, PrecisionNode)
        assert isinstance(symbol, str)

        rhs_error = rhs.error.copy()
        for (i, error) in enumerate(rhs_error):
            rhs_error[i].value *= self.val

        lhs_error = self.error.copy()
        for (i, error) in enumerate(lhs_error):
            lhs_error[i].value *= rhs.val

        total_err = lhs_error + rhs_error

        # Store multiplication of errors as a tuple whose first elemeent is E_x and second element is E_y
        return PrecisionNode(self.val * rhs.val, symbol, total_err.append((rhs.error, self.error)))
