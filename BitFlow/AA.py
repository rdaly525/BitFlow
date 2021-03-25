from .IA import Interval
from copy import deepcopy


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
        noise_str = " + ".join((f"{v}*eps{k}" for k,
                                v in sorted(self.noise.items())))
        return f"{self.base} + {noise_str}"

    def __eq__(self, rhs):
        return rhs.base == rhs.base and rhs.noise == rhs.noise

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
            return AInterval(self.base + rhs, self.noise)

    def __radd__(self, rhs):
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
            return AInterval(self.base + rhs, self.noise)

    def __sub__(self, rhs):
        if isinstance(rhs, AInterval):
            new_base = self.base - rhs.base
            new_noise = self.noise
            es = set(rhs.noise.keys())
            for e in es:
                noise_val = 0
                for noise in (self.noise, rhs.noise):
                    if e in noise:
                        noise_val -= noise[e]
                new_noise[e] = noise_val
            return AInterval(new_base, new_noise)
        else:
            return AInterval(self.base - rhs, self.noise)

    def __rsub__(self, rhs):
        if isinstance(rhs, AInterval):
            new_base = self.base - rhs.base
            new_noise = self.noise
            es = set(rhs.noise.keys())
            for e in es:
                noise_val = 0
                for noise in (self.noise, rhs.noise):
                    if e in noise:
                        noise_val -= noise[e]
                new_noise[e] = noise_val
            return AInterval(new_base, new_noise)
        else:
            return AInterval(self.base - rhs, self.noise)

    def __mul__(self, rhs):
        if isinstance(rhs, AInterval):
            x_sum = 0
            y_sum = 0

            es1 = self.noise.copy()
            for key in es1:
                x_sum += abs(es1[key])
                es1[key] *= rhs.base

            es2 = rhs.noise.copy()
            for key in es2:
                y_sum += abs(es2[key])
                es2[key] *= self.base

            new_noise = self.merge_dicts(es1, es2)
            new_noise[max(set(new_noise.keys())) + 1] = x_sum * y_sum
            return AInterval(self.base * rhs.base, new_noise)
        else:
            this = deepcopy(self)
            for key in this.noise:
                this.noise[key] *= rhs
            return AInterval(self.base * rhs, this.noise)

    def __rmul__(self, rhs):
        if isinstance(rhs, AInterval):
            x_sum = 0
            y_sum = 0

            es1 = self.noise.copy()
            for key in es1:
                x_sum += abs(es1[key])
                es1[key] *= rhs.base

            es2 = rhs.noise.copy()
            for key in es2:
                y_sum += abs(es2[key])
                es2[key] *= self.base

            new_noise = self.merge_dicts(es1, es2)
            new_noise[max(set(new_noise.keys())) + 1] = x_sum * y_sum
            return AInterval(self.base * rhs.base, new_noise)
        else:
            this = deepcopy(self)
            for key in this.noise:
                this.noise[key] *= rhs
            return AInterval(self.base * rhs, this.noise)

    def merge_dicts(self, dict1, dict2):
        new_dict = dict1
        for key in set(dict2.keys()):
            if key in new_dict:
                new_dict[key] += dict2[key]
            else:
                new_dict[key] = dict2[key]
        return new_dict
