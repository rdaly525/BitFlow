from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Select, Output


class caseStudy:

    def poly_approx():
        a = Input(name="a")
        c = Input(name="c")

        first = a[0]

        for i in range(1, 5):  # for degree 4 polynomial
            first = first * c + a[i]

        casestudy_dag = Dag(outputs=[first], inputs=[a, c])

        return casestudy_dag

    def RGB_to_YCbCr():
        a = Input(name="a")

        col_1 = Constant(.299) * a[0] + Constant(.587) * \
            a[1] + Constant(.114) * a[2]
        col_2 = Constant(-.16875) * \
            a[0] + Constant(-.33126) * a[1] + Constant(.5) * a[2]
        col_3 = Constant(.5) * a[0] + Constant(-.41869) * \
            a[1] + Constant(-.08131) * a[2]

        casestudy_dag = Dag(outputs=[col_1, col_2, col_3], inputs=[a])

        return casestudy_dag

    def Matrix_Multiplication():
        a = Input(name="a")
        b = Input(name="b")

        a00 = a[0][0]
        a01 = a[0][1]
        a10 = a[1][0]
        a11 = a[1][1]

        b00 = b[0][0]
        b01 = b[0][1]
        b10 = b[1][0]
        b11 = b[1][1]

        p0 = (a00 + a11) * (b00 + b11)
        p1 = (a10 + a11) * b00
        p2 = a00 * (b01 - b11)
        p3 = a11 * (b10 - b00)
        p4 = (a00 + a01) * b11
        p5 = (a10 - a00) * (b00 + b01)
        p6 = (a01 - a11) * (b10 + b11)

        y00 = p0 + p3 - p4 + p6
        y01 = p2 + p4
        y10 = p1 + p3
        y11 = p0 + p2 - p1 + p5

        casestudy_dag = Dag(outputs=[y00, y01, y10, y11], inputs=[a, b])
        return casestudy_dag
