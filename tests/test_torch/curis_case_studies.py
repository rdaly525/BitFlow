

def poly_approx(coefArray,n,x):#n elements



    first=coefArray[0]

    for i in range(1,n):
        first=first*x+coefArray[i]

    return first

def RGB_to_YCbCr(RGB):#assuming RGB matrix passed in as three row, 1 column
    matrixA = np.array([[.299, .587, .114], [-.16875, -.33126, .5], [.5, -.41869, -.08131]])
    retYCbCr=RGB.dot(matrixA*RGB)

    return retYCbCr


def Matrix_Multiplication(first,second):

    a00=first[0,0]
    a01=first[0,1]
    a10=first[1,0]
    a11=first[1,1]

    b00=second[0,0]
    b01=second[0,1]
    b10=second[1,0]
    b11=second[1,1]

    p0 =( a00 + a11)(b00 + b11)
    p1 =( a10 + a11)b00
    p2 = a00(b01 −b11)
    p3 = a11(b10 −b00)
    p4 =( a00 + a01)b11
    p5 =( a10 −a00)(b00 + b01)
    p6 =( a01 −a11)(b10 + b11).



    y00 = p0 + p3 −p4 + p6
    y01 = p2 + p4
    y10 = p1 + p3
    y11 = p0 + p2 −p1 + p5

    toRet=[[y00,y01],[y10,y11]]
    return toRet
