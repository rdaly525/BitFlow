from Precision import PrecisionNode, FPEpsilon, FPEpsilonMultiplier

def test_add():
    x = PrecisionNode(5, "x", [])
    y = PrecisionNode(10, "y", [])
    z = x.add(y, "z")
    assert z == PrecisionNode(15, "z", [FPEpsilon("x"), FPEpsilon("y")])

def test_sub():
    x = PrecisionNode(20, "x", [])
    y = PrecisionNode(12, "y", [])
    z = x.sub(y, "z")
    assert z == PrecisionNode(8, "z", [FPEpsilon("x"), FPEpsilon("y", -1)])

def test_mul():
    x = PrecisionNode(3, "x", [])
    y = PrecisionNode(4, "y", [])
    z = x.mul(y, "z")
    assert z == PrecisionNode(12, "z", [FPEpsilon("x", 4), FPEpsilon("y", 3), FPEpsilonMultiplier([FPEpsilon("x")], [FPEpsilon("x")])])

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

test_paper_fig3()
test_add()
test_sub()
test_mul()