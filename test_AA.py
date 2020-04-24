from AA import AInterval
from IA import Interval

def test_to_string():
    #Two different wyas to construct an AInterval

    eps_idx = 1
    lo, hi = 0.0, 6.0
    x = AInterval(lo, hi, eps_idx=eps_idx)
    assert str(x) == "3.0 + 3.0*eps1"

    base = 3.0
    noise = {eps_idx: 3.0}
    x = AInterval(3.0, {1: 3.0})
    assert str(x) == "3.0 + 3.0*eps1"

    #AInterval can be converted to an I interval
    assert x.to_interval() == Interval(0.0, 6.0)

def test_add():
    x = AInterval(0.0, 6.0, eps_idx=1)
    y = AInterval(2.0, 8.0, eps_idx=2)
    z = x + y
    assert z == AInterval(8.0, {1: 3.0, 2: 3.0})

def test_sub():
    x = AInterval(-2.0, 6.0, eps_idx=1)
    y = AInterval(2.0, 10.0, eps_idx=2)
    z = x - y
    #assert z == TODO

def test_mul():
    x = AInterval(-2.0, 6.0, eps_idx=1)
    y = AInterval(2.0, 10.0, eps_idx=2)
    z = x * y
    #assert z == TODO

def test_add_const():
    x = AInterval(0.0, 6.0, eps_idx=1)
    y = 5.0
    z = x - y
    #assert z == TODO

def test_sub_const():
    x = AInterval(0.0, 6.0, eps_idx=1)
    y = 5.0
    z = x - y
    #assert z == TODO

def test_mul_const():
    x = AInterval(0.0, 6.0, eps_idx=1)
    y = 5.0
    z = x * y
    #assert z == TODO

def test_paper_fig3():
    a = AInterval(-3.0, 2.0, eps_idx=1)
    b = AInterval(4.0, 8.0, eps_idx=2)
    c = 4.3
    d = a*b
    e = d+c
    z = e-b

    assert d.to_interval() == Interval(-24.0, 18.0)
    assert e.to_interval() == Interval(-19.7, 22.3)
    assert z.to_interval() == Interval(-25.7, 16.3)

