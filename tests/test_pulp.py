import pulp

def test_pulp():
    Var = lambda name: pulp.LpVariable(name, cat='Integer')
    x = Var('x')
    y = Var('y')

    p = pulp.LpProblem("My minimization problem", pulp.LpMinimize)

    #Objective (What I am minimizing)
    p += (4*x + 3*y, "Z")

    #Constraints
    p += (2*y <= 25-x)
    p += (4*y >= 2*x-8)
    p += (y <= 2*x-5)
    p += (y >= 0)
    p += (x >= 0)

    p.solve()
    solved = p.status
    assert solved

    expected_x = 3
    expected_y = 0

    assert x.varValue == expected_x
    assert y.varValue == expected_y
