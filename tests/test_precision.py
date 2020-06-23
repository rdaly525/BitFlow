from BitFlow.Precision import PrecisionNode, FPEpsilon, FPEpsilonMultiplier

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
    b = PrecisionNode(3, "b", [])
    c = PrecisionNode(4, "c", [])

    d = a.mul(b, "d")
    e = d.add(c, "e")
    z = e.sub(b, "z")

    assert z == PrecisionNode(7, "z", [FPEpsilon("a", 3), FPEpsilon("b", 2), FPEpsilonMultiplier([FPEpsilon("b")], [FPEpsilon("a")]), FPEpsilon("d"), FPEpsilon("c"), FPEpsilon("e"), FPEpsilon("b", -1)])
