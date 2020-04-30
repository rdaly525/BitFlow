from curis.IA import Interval

def test_add():
    x = Interval(0, 5)
    y = Interval(3, 8)
    z = x + y
    print(z)
    assert z == Interval(3, 13)

def test_sub():
    x = Interval(0, 5)
    y = Interval(3, 8)
    z = x - y
    print(z)
    assert z == Interval(-8, 2)

def test_mul():
    x = Interval(-2, 5)
    y = Interval(3, 8)
    z = x*y
    print(z)
    assert z == Interval(-16, 40)

def test_add_const():
    x = Interval(0, 5)
    y = 10
    z = x + y
    print(z)
    assert z == Interval(10, 15)

def test_sub_const():
    x = Interval(0, 5)
    y = 10
    z = x - y
    print(z)
    assert z == Interval(-10, -5)

def test_mul_const():
    x = Interval(0, 5)
    y = 10
    z = x * y
    print(z)
    assert z == Interval(0, 50)

def test_self_sub():
    x = Interval(-5,5)
    z = x-x
    assert z == Interval(-10, 10)
