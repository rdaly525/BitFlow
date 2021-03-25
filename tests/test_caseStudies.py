from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval


def test_poly_approx():

    fig_casestudy = caseStudy.poly_approx()
    evaluator = NumEval(fig_casestudy)

    a, c = [1, 3, -6, -10, -1], 3

    res = evaluator.eval(a=a, c=c)
    gold = 77
    assert res == gold


def test_RGB_to_YCbCr():

    fig_casestudy = caseStudy.RGB_to_YCbCr()
    evaluator = NumEval(fig_casestudy)

    a = [22, 103, 200]

    res = evaluator.eval(a=a)

    gold = [89.839, 62.16772, -48.38707]
    assert res == gold


def test_Matrix_Multiplication():

    fig_casestudy = caseStudy.Matrix_Multiplication()
    evaluator = NumEval(fig_casestudy)

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]

    print("matrix")
    print(a)
    res = evaluator.eval(a=a, b= b)
    print(res)
    gold = [19, 22, 43, 50]
    assert res == gold
