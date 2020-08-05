from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch
from BitFlow.utils import Imgs2Dataset

from torch.utils import data
import time


# def test_image2data():
#     converter = Imgs2Dataset()
#     dataset, size = converter.convert_rgb2ycbcr('./data/imgs/')
#     train_set, test_set = torch.utils.data.random_split(
#         dataset, [int(round(0.8 * size)), int(round(0.2 * size))])

#     return


def gen_fig3():
    # (a*b) + 4 - b
    a = Input(name="a")
    b = Input(name="b")
    c = Constant(4.3, name="c")
    d = Mul(a, b, name="d")
    e = Add(d, c, name="e")
    z = Sub(e, b, name="z")

    fig3_dag = Dag(outputs=[z], inputs=[a, b])
    return fig3_dag


def gen_ex1():
    # (a * b) + (b * c)
    a = Input(name="a")
    b = Input(name="b")
    c = Input(name="c")
    d = Mul(a, b, name="d")
    e = Mul(b, c, name="e")
    z_1 = Add(e, d, name="z_1")
    z_2 = Add(a, d, name="z_2")

    dag = Dag(outputs=[z_1, z_2], inputs=[a, b, c])
    return dag


# def test_fig3():

#     t0 = time.time()

#     dag = gen_fig3()

#     bf = BitFlow(dag, {"z": 8.}, {'a': (-3., 2.),
#                                   'b': (4., 8.)}, lr=5e-4, range_lr=1e-4, train_range=True, training_size=10000, testing_size=2000, distribution=0)
#     bf.train(epochs=100)

#     # # check saving object works
#     # BitFlow.save("./models/fig3", bf)
#     # new_bf = BitFlow.load("./models/fig3")

#     # new_bf.train(epochs=5)

#     # assert new_bf.range_lr == bf.range_lr

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     return


# def test_ex1():
#     t0 = time.time()

#     dag = gen_ex1()

#     bf = BitFlow(dag, {"z_1": 5., "z_2": 8.}, {
#         'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)}, lr=4e-4, range_lr=1e-4, train_range=False, training_size=10000, testing_size=2000)
#     bf.train(epochs=30)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")
#     return


def RGB_to_YCbCr():
    r = Input(name="r")
    g = Input(name="g")
    b = Input(name="b")

    col_1 = Add(Add(Mul(Constant(.299, name="C1"), r), Mul(Constant(.587, name="C2"), g)),
                Mul(Constant(.114, name="C3"), b), name="col_1")
    col_2 = Add(Add(Mul(Constant(-.16875, name="C4"), r), Mul(Constant(-.33126, name="C5"), g)),
                Mul(Constant(.5, name="C6"), b), name="col_2")
    col_3 = Add(Add(Mul(Constant(.5, name="C7"), r), Mul(Constant(-.41869, name="C8"), g)),
                Mul(Constant(-.08131, name="C9"), b), name="col_3")

    casestudy_dag = Dag(outputs=[col_1, col_2, col_3], inputs=[r, g, b])

    return casestudy_dag


# def test_rgb_case_study():
#     t0 = time.time()

#     dag = RGB_to_YCbCr()

#     params = dict(
#         training_size=10000,
#         testing_size=2000,
#         batch_size=16,
#         lr=1e-4,
#         train_range=True,
#         range_lr=5e-5,
#         distribution=0
#     )

#     bf = BitFlow(dag, {"col_1": 12., "col_2": 12., "col_3": 12.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=20)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     # Sample Matrix Product
#     test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return


# def test_rgb_case_study_custom_dataset():
#     dag = RGB_to_YCbCr()

#     converter = Imgs2Dataset()
#     dataset, size = converter.convert_rgb2ycbcr('./data/imgs/')
#     train_set, test_set = torch.utils.data.random_split(
#         dataset, [int(round(0.8 * size)), int(round(0.2 * size))])

#     ds_params = dict(
#         batch_size=16
#     )

#     train_gen = data.DataLoader(train_set, **ds_params)
#     test_gen = data.DataLoader(test_set, **ds_params)

#     params = dict(
#         training_size=int(round(0.8 * size)),
#         testing_size=int(round(0.2 * size)),
#         batch_size=16,
#         lr=8e-4,
#         train_range=False,
#         range_lr=5e-4,
#         distribution=0,
#         custom_data=(train_gen, test_gen),
#         test_optimizer=False,
#         test_ufb=True
#     )

#     bf = BitFlow(dag, {"col_1": 8., "col_2": 8., "col_3": 8.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=5)

#     # Sample Matrix Product
#     test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

def matrix_multiply():
    a00 = Input(name="a00")
    a01 = Input(name="a01")
    a10 = Input(name="a10")
    a11 = Input(name="a11")

    b00 = Input(name="b00")
    b01 = Input(name="b01")
    b10 = Input(name="b10")
    b11 = Input(name="b11")

    C = Constant(-1., name="C")

    p0 = Mul(Add(a00, a11), Add(b00, b11))
    p1 = Mul(Add(a10, a11), b00)
    p2 = Mul(a00, Add(b01, Mul(C, b11)))
    p3 = Mul(a11, Add(b10, Mul(C, b00)))
    p4 = Mul(b11, Add(a00, a01))
    p5 = Mul(Add(b00, b01), Add(a10, Mul(C, a00)))
    p6 = Mul(Add(b10, b11), Add(a01, Mul(C, a11)))

    y00 = Add(Add(Add(p0, p3), Mul(C, p4)), p6, name="y00")
    y01 = Add(p2, p4, name="y01")
    y10 = Add(p1, p3, name="y10")
    y11 = Add(Add(Add(p0, p2), Mul(C, p1)), p5, name="y11")

    matrix_dag = Dag(outputs=[y00, y01, y10, y11], inputs=[
        a00, a01, a10, a11, b00, b01, b10, b11])

    return matrix_dag


def test_matrix_case_study():
    t0 = time.time()

    dag = matrix_multiply()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=5e-4,
        train_range=True,
        range_lr=1e-4,
        distribution=0,
        test_optimizer=False
    )

    bf = BitFlow(dag, {"y00": 8., "y01": 8., "y10": 8., "y11": 8.}, {
        'a00': (-11., 10.), 'a01': (-10., 12.), 'a10': (-13., 10.), 'a11': (-14., 10.), 'b00': (-11., 10.), 'b01': (-10., 12.), 'b10': (-13., 10.), 'b11': (-14., 10.)}, **params)
    bf.train(epochs=20)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    # Sample Matrix Product (Identity Product)
    test = {"a00": 1., "a01": 1., "a10": 1., "a11": 1., "b00": 1.,
            "b01": 0., "b10": 0., "b11": 1., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return
