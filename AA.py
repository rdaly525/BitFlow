from IA import Interval

class AInterval:
    def __init__(self, arg0, arg1, *, eps_idx=None):
        if eps_idx is not None:
            assert isinstance(arg0, (float, int))
            assert isinstance(arg1, (float, int))
            lo, hi = arg0, arg1
            self.base = (hi + lo)/2
            self.noise = {eps_idx: (hi - lo)/2}
        else:
            assert isinstance(arg0, (float, int))
            isinstance(arg1, dict)
            self.base = arg0
            self.noise = arg1

    def __str__(self):
        noise_str = " + ".join(( f"{v}*eps{k}" for k,v in sorted(self.noise.items())))
        return f"{self.base} + {noise_str}"

    def __eq__(self, rhs):
        return rhs.base == rhs.base and rhs.noise==rhs.noise

    def to_interval(self):
        lo, hi = self.base, self.base
        for e, v in self.noise.items():
            lo -= abs(v)
            hi += abs(v)
        return Interval(lo, hi)

    def __add__(self, rhs):
        if isinstance(rhs, AInterval):
            new_base = self.base + rhs.base
            new_noise = {}
            es = set(self.noise.keys()).union(rhs.noise.keys())
            for e in es:
                noise_val = 0
                for noise in (self.noise, rhs.noise):
                    if e in noise:
                        noise_val += noise[e]
                new_noise[e] = noise_val
            return AInterval(new_base, new_noise)
        else:
            raise NotImplementedError("TODO: adding a constant")

    def __sub__(self, rhs):
        raise NotImplementedError("TODO: subtract AInterval or a constant")

    def __mul__(self, rhs):
        raise NotImplementedError("TODO: multiply")


