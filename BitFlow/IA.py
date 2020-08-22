from itertools import product


class Interval:
    def __init__(self, lo, hi):
        assert lo <= hi, f"{lo} must be less than {hi}"
        self.lo = lo
        self.hi = hi

    def __add__(self, rhs):
        if isinstance(rhs, Interval):
            return Interval(self.lo + rhs.lo, self.hi + rhs.hi)
        else:
            assert isinstance(rhs, float)
            return Interval(self.lo + rhs, self.hi + rhs)

    def __sub__(self, rhs):
        if isinstance(rhs, Interval):
            lo = self.lo - rhs.hi
            hi = self.hi - rhs.lo
            return Interval(lo, hi)
        else:
            return Interval(self.lo - rhs, self.hi - rhs)

    def __mul__(self, rhs):
        if isinstance(rhs, Interval):
            poss = list(map(lambda x: x[0]*x[1],
                            product(self.interval, rhs.interval)))
            return Interval(min(poss), max(poss))
        else:
            return Interval(self.lo * rhs, self.hi * rhs)

    def __eq__(self, rhs):
        assert isinstance(rhs, Interval)
        return self.lo == rhs.lo and self.hi == rhs.hi

    def __str__(self):
        return f"[{self.lo}, {self.hi}]"

    @property
    def interval(self):
        return [self.lo, self.hi]
