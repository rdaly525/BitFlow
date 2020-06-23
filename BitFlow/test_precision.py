from Precision import PrecisionNode, FPEpsilon

# def test_to_string():
#     #Two different wyas to construct an AInterval
#
#     eps_idx = 1
#     lo, hi = 0.0, 6.0
#     x = AInterval(lo, hi, eps_idx=eps_idx)
#     assert str(x) == "3.0 + 3.0*eps1"
#
#     base = 3.0
#     noise = {eps_idx: 3.0}
#     x = AInterval(3.0, {1: 3.0})
#     assert str(x) == "3.0 + 3.0*eps1"
#
#     #AInterval can be converted to an I interval
#     assert x.to_interval() == Interval(0.0, 6.0)
#
# def test_add():
#     x = AInterval(0.0, 6.0, eps_idx=1)
#     y = AInterval(2.0, 8.0, eps_idx=2)
#     z = x + y
#     assert z == AInterval(8.0, {1: 3.0, 2: 3.0})
#
# def test_sub():
#     x = AInterval(-2.0, 6.0, eps_idx=1)
#     y = AInterval(2.0, 10.0, eps_idx=2)
#     z = x - y
#     assert z == AInterval(-4.0, {1: 4.0, 2: -4.0})
#
# def test_mul():
#     x = AInterval(-2.0, 6.0, eps_idx=1)
#     y = AInterval(2.0, 10.0, eps_idx=2)
#     z = x * y
#     assert z == AInterval(12.0, {1: 24, 2: 8, 3: 16})
#
# def test_add_const():
#     x = AInterval(0.0, 6.0, eps_idx=1)
#     y = 5.0
#     z = x + y
#     assert z == AInterval(8.0, {1: 3})
#
# def test_sub_const():
#     x = AInterval(0.0, 6.0, eps_idx=1)
#     y = 5.0
#     z = x - y
#     assert z == AInterval(-2.0, {1: 3})
#
# def test_mul_const():
#     x = AInterval(0.0, 6.0, eps_idx=1)
#     y = 5.0
#     z = x * y
#     assert z == AInterval(15, {1: 15})

def test_paper_fig3():
    a = PrecisionNode(2, "a", [])
    print(a)
    b = PrecisionNode(3, "b", [])
    print(b)
    c = PrecisionNode(4, "c", [])
    print(c)
    d = a.mul(b, "d")
    print(d)
    e = d.add(c, "e")
    print(e)
    z = e.sub(b, "z")
    print(z)

test_paper_fig3();
