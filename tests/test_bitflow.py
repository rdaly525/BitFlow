from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, LookupTable, Select, BitShift, Concat, \
    Reduce
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
from BitFlow.MNIST.MNIST_library import linear_layer, matrix_multiply, dot_product
import torch
from BitFlow.utils import Imgs2Dataset

from torch.utils import data
import time

import numpy as np


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


def test_fig3():
    t0 = time.time()

    dag = gen_fig3()

    bf = BitFlow(dag, {"z": 8.}, {'a': (-3., 2.),
                                  'b': (4., 8.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=10000,
                 testing_size=2000, distribution=0, incorporate_ulp_loss=False, batch_size=16, test_optimizer=True)
    bf.train(epochs=15)

    # check saving object works
    BitFlow.save("./models/fig3", bf)
    new_bf = BitFlow.load("./models/fig3")

    new_bf.train(epochs=5)

    assert new_bf.range_lr == bf.range_lr

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    return


def test_ex1():
    t0 = time.time()

    dag = gen_ex1()

    bf = BitFlow(dag, {"z_1": 8., "z_2": 8.}, {
        'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=10000,
                 testing_size=2000, incorporate_ulp_loss=False, test_optimizer=True)
    bf.train(epochs=15)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")
    return


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


def test_rgb_case_study():
    print("\n=== RGB ===")
    t0 = time.time()

    dag = RGB_to_YCbCr()

    params = dict(
        training_size=1000,
        testing_size=200,
        batch_size=16,
        lr=4e-3,
        train_range=True,
        range_lr=4e-3,
        distribution=0,
        test_optimizer=True,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"col_1": 8., "col_2": 8., "col_3": 8.}, {
        'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
    bf.train(epochs=10)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    # Sample Matrix Product
    test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


def test_rgb_case_study_custom_dataset():
    dag = RGB_to_YCbCr()

    converter = Imgs2Dataset()
    dataset, size = converter.convert_rgb2ycbcr('./data/imgs/')
    train_set, test_set = torch.utils.data.random_split(
        dataset, [int(round(0.8 * size)), int(round(0.2 * size))])

    ds_params = dict(
        batch_size=16
    )

    train_gen = data.DataLoader(train_set, **ds_params)
    test_gen = data.DataLoader(test_set, **ds_params)

    params = dict(
        training_size=int(round(0.8 * size)),
        testing_size=int(round(0.2 * size)),
        batch_size=16,
        lr=1e-4,
        train_range=True,
        range_lr=1e-4,
        distribution=0,
        custom_data=(train_gen, test_gen),
        test_optimizer=False,
        test_ufb=True
    )

    bf = BitFlow(dag, {"col_1": 8., "col_2": 8., "col_3": 8.}, {
        'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
    bf.train(epochs=1)

    # Sample Matrix Product
    test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


# def matrix_multiply():
#     a00 = Input(name="a00")
#     a01 = Input(name="a01")
#     a10 = Input(name="a10")
#     a11 = Input(name="a11")
#
#     b00 = Input(name="b00")
#     b01 = Input(name="b01")
#     b10 = Input(name="b10")
#     b11 = Input(name="b11")
#
#     C = Constant(-1., name="C")
#
#     p0 = Mul(Add(a00, a11), Add(b00, b11))
#     p1 = Mul(Add(a10, a11), b00)
#     p2 = Mul(a00, Add(b01, Mul(C, b11)))
#     p3 = Mul(a11, Add(b10, Mul(C, b00)))
#     p4 = Mul(b11, Add(a00, a01))
#     p5 = Mul(Add(b00, b01), Add(a10, Mul(C, a00)))
#     p6 = Mul(Add(b10, b11), Add(a01, Mul(C, a11)))
#
#     y00 = Add(Add(Add(p0, p3), Mul(C, p4)), p6, name="y00")
#     y01 = Add(p2, p4, name="y01")
#     y10 = Add(p1, p3, name="y10")
#     y11 = Add(Add(Add(p0, p2), Mul(C, p1)), p5, name="y11")
#
#     matrix_dag = Dag(outputs=[y00, y01, y10, y11], inputs=[
#         a00, a01, a10, a11, b00, b01, b10, b11])
#
#     return matrix_dag


def test_matrix_case_study():
    print("\n=== MATRIX ===")
    t0 = time.time()

    dag = matrix_multiply()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=8,
        lr=4e-3,
        train_range=True,
        range_lr=4e-3,
        distribution=0,
        test_optimizer=True,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"y00": 8., "y01": 8., "y10": 8., "y11": 8.}, {
        'a00': (-10., 10.), 'a01': (-10., 10.), 'a10': (-10., 10.), 'a11': (-10., 10.), 'b00': (-10., 10.),
        'b01': (-10., 10.), 'b10': (-10., 10.), 'b11': (-10., 10.)}, **params)
    bf.train(epochs=0)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    # Sample Matrix Product (Identity Product)
    test = {"a00": 1., "a01": 1., "a10": 1., "a11": 1., "b00": 1.,
            "b01": 0., "b10": 0., "b11": 1., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


def generate_poly_approx(a, b, c, d, e):
    x = Input(name="x")
    a = Constant(a, name="a")
    b = Constant(b, name="b")
    c = Constant(c, name="c")
    d = Constant(d, name="d")
    e = Constant(e, name="e")

    output = Add(
        Mul(Add(Mul(Add(Mul(Add(Mul(a, x), b), x), c), x), d), x), e, name="res")

    approx_dag = Dag(outputs=[output], inputs=[x])

    return approx_dag


def test_poly_approx():
    print("\n=== POLY APPROX ===")
    t0 = time.time()

    dag = generate_poly_approx(-0.25, 1. / 3, -0.5, 1., 0.)

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=4e-3,
        train_range=True,
        range_lr=4e-3,
        distribution=0,
        test_optimizer=True,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"res": 8.}, {"x": (0., 1.)}, **params)
    bf.train(epochs=5)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    test = {"x": 0.3, "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


def generate_basic_lut():
    x = Input(name="x")
    amplitude = Input(name="a")
    shift = Input(name="b")

    output = Add(Mul(amplitude, LookupTable(
        np.sin, x)), shift, name="res")
    # output = Add(Mul(amplitude, x), shift, name="res")

    sin_dag = Dag(outputs=[output], inputs=[x, amplitude, shift])

    return sin_dag


def test_basic_lut():
    t0 = time.time()

    dag = generate_basic_lut()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=True,
        incorporate_ulp_loss=False
    )

    bf = BitFlow(dag, {"res": 8.}, {"x": (-2., 2.),
                                    "a": (1., 5.), "b": (-3, 3)}, **params)
    bf.train(epochs=20)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    test = {"x": 0.3, "a": 2., "b": 1., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


def generate_square_wave():
    x = Input(name="x")

    a = LookupTable(np.sin, x)
    b = Add(Mul(Constant(1 / 3.), LookupTable(np.sin, Mul(Constant(3.), x))), a)
    c = Add(Mul(Constant(1 / 5.), LookupTable(np.sin, Mul(Constant(5.), x))), b)
    d = Add(Mul(Constant(1 / 7.), LookupTable(np.sin, Mul(Constant(7.), x))), c)
    e = Add(Mul(Constant(1 / 9.), LookupTable(np.sin,
                                              Mul(Constant(9.), x))), d, name="res")

    dag = Dag(outputs=[e], inputs=[x])

    return dag


def test_square_wave():
    print("=== SQUARE WAVE ===")
    t0 = time.time()

    dag = generate_square_wave()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-2,
        train_range=True,
        range_lr=1e-2,
        distribution=0,
        test_optimizer=True,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"res": 8.}, {"x": (-3., 3.)}, **params)
    bf.train(epochs=10)

    print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

    test = {"x": 2.1, "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return


def generate_vector_example2():
    X = Input(name="X")
    weight = Input(name="weight")

    # out = Reduce(Concat(Select(Select(X, 0), 0), Select(
    #     Select(X, 1), 0), concat_dim=0), reduce_dim=0, name="res")

    out = matrix_multiply(X, weight, 10, 1, 1)

    dag = Dag(outputs=[out], inputs=[X, weight])

    return dag


def test_vector_ex2():
    dag = generate_vector_example2()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"res": 4.}, {"X": torch.tensor([[(0., 1.) for i in range(10)] for j in range(1)]),
                                    "weight": torch.tensor([[(0., 1.) for i in range(1)] for j in range(1)])}, **params)

    # bf = BitFlow(dag, {"res": 4.}, {"X": [[(0., 1.) for i in range(10)] for j in range(7)],"weight": [[(0., 1.) for i in range(7)] for j in range(10)]}, **params)
    bf.train(epochs=5)

    return


def generate_vector_example():
    X = Input(name="X")

    out = Reduce(Concat(X[0][0], X[1][0], concat_dim=0), reduce_dim=0, name="res")

    dag = Dag(outputs=[out], inputs=[X])

    return dag


def test_vector_ex():
    dag = generate_vector_example()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"res": 4.}, {"X": [(0., 1.), (0., 1.)]}, **params)
    bf.train(epochs=10)

    return

def generate_vector_example5():
    X = Input(name="X")

    out = Select(X[0],0,name="out")

    dag = Dag(outputs=[out], inputs=[X])

    return dag


def test_vector_ex5():
    dag = generate_vector_example5()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"out": 4.}, {"X": [[(0., 1.)], [(0., 1.)]]}, **params)
    bf.train(epochs=10)

    return


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    weight = Input(name="weight")
    bias = Input(name="bias")
    # _concat_add__concat_bias = linear_layer(X, weight, bias, row, col, size)
    _concat_add_final_bias = linear_layer(X, weight, bias, row, col, size)

    fig = Dag(outputs=[_concat_add_final_bias], inputs=[X, weight, bias])
    return fig


def test_linearlayer():
    row = 100
    col = 10
    size = 784
    dag = gen_linearlayer(row, col, size)

    params = dict(
        training_size=60000,
        testing_size=2000,
        batch_size=100,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    # bf = BitFlow(dag, {"y": [(0., 1.) for i in range(10)]}, {"X": [[(0., 1.) for i in range(row)] for j in range(size)],
    #                                                          "weight": [[(0., 1.) for i in range(row)] for j in range(size)],
    #                                                          "bias": [(0., 1.) for i in range(col)]}, **params)

    bf = BitFlow(dag, {"_concat_add_final_bias": torch.ones(10).fill_(1)}, {"X": torch.ones(row, size).fill_(1),
                                                       "weight": torch.ones(size, col).fill_(1),
                                                       "bias": torch.ones(col).fill_(1)}, **params)
    bf.train(epochs=5)

    return


def generate_vector_example3():
    X = Input(name="X")
    W = Input(name="weight")

    out = Mul(X, W, name="out")

    dag = Dag(outputs=[out], inputs=[X])

    return dag


def test_vector_ex3():
    dag = generate_vector_example3()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"out": 4.}, {"X": torch.tensor([1., 1.]), "weight": torch.tensor([1., 1.])}, **params)
    bf.train(epochs=10)

    return


def generate_vector_example6():
    X = Input(name="X")
    W = Input(name="weight")

    out = Add(X, W, name="out")

    dag = Dag(outputs=[out], inputs=[X])

    return dag


def test_vector_ex6():
    dag = generate_vector_example6()

    params = dict(
        training_size=10000,
        testing_size=2000,
        batch_size=16,
        lr=1e-3,
        train_range=True,
        range_lr=1e-3,
        distribution=0,
        test_optimizer=False,
        incorporate_ulp_loss=True
    )

    bf = BitFlow(dag, {"out": 4.}, {"X": torch.tensor([1., 1.]), "weight": torch.tensor([1., 1.])}, **params)
    bf.train(epochs=10)

    return



#test_vector_ex5() WORKS
#test_vector_ex() WORKS

#test_vector_ex6()
test_linearlayer()