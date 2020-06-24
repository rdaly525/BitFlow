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

# a more complicated test case with multiple multiply nodes
def test_example_1():
    x = PrecisionNode(2, "x", [])
    y = PrecisionNode(5, "y", [])
    z = PrecisionNode(3, "z", [])

    i = x.mul(y, "i")
    j = y.sub(z, "j")
    k = i.mul(j, "k")

    assert k == PrecisionNode(20, "k", [FPEpsilon("y", 10), FPEpsilon("z", -10), FPEpsilon("j", 10), FPEpsilon("y", 4), FPEpsilon("x", 10), FPEpsilonMultiplier([FPEpsilon("y")], [FPEpsilon("x")], 2), FPEpsilon("i", 2), FPEpsilonMultiplier([FPEpsilon("y"), FPEpsilon("z", -1), FPEpsilon("j")], [FPEpsilon("y", 2), FPEpsilon("x", 5), FPEpsilonMultiplier([FPEpsilon("y")], [FPEpsilon("x")]), FPEpsilon("i")])])

def test_example_2():
    a = PrecisionNode(10, "a", [])
    b = PrecisionNode(8, "b", [])

    c = a.add(b, "c")
    d = c.sub(a, "d")
    e = d.add(b, "e")

    assert e == PrecisionNode(16, "e", [FPEpsilon("a"), FPEpsilon("b"), FPEpsilon("c"), FPEpsilon("a", -1), FPEpsilon("b"),  FPEpsilon("d")])


# def test_paper_fig3():
#     a = PrecisionNode(2, "a", [])
#     print(a)
#     b = PrecisionNode(3, "b", [])
#     print(b)
#     c = PrecisionNode(4, "c", [])
#     print(c)
#     d = a.mul(b, "d")
#     print(d)
#     e = d.add(c, "e")
#     print(e)
#     z = e.sub(b, "z")
#     print(z)
#
# test_paper_fig3()
