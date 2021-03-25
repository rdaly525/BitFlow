from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, LookupTable, Select, BitShift, Concat, Reduce
from BitFlow.BitFlow2Verilog import BitFlow2Verilog
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch
from BitFlow.utils import Imgs2Dataset

from torch.utils import data
import time

import numpy as np


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


def gen_ex2():

    # sqrt((a * a) + (b * b) + (c * c))

    a = Input(name="a")
    b = Input(name="b")
    c = Input(name="c")

    out = LookupTable(np.sqrt, Add(
        Add(Mul(a, a), Mul(b, b)), Mul(c, c)), name="res", precision=8.)

    dag = Dag(outputs=[out], inputs=[a, b, c])
    return dag


def gen_ex3():

    a = Input(name="a")
    b = Input(name="b")
    c = Input(name="c")
    d = Input(name="d")

    out = Add(Mul(a, Mul(b, Mul(c, d))), Mul(a, b), name="res")

    dag = Dag(outputs=[out], inputs=[a, b, c, d])
    return dag


# def test_fig3():

#     dag = gen_fig3()

#     bf = BitFlow(dag, {"z": 8.}, {'a': (-3., 2.),
#                                   'b': (4., 8.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=50000, testing_size=10000, distribution=2, incorporate_ulp_loss=False, batch_size=16, test_optimizer=True)
#     bf.train(epochs=10, decay=0.8)

#     rounded_dag = bf.rounded_dag
#     original_dag = bf.original_dag

#     verilog_gen = BitFlow2Verilog(
#         "fig3", bf.P, bf.R, bf.filtered_vars, original_dag, {"z": 8.})
#     verilog_gen.evaluate()

    # # check saving object works
    # BitFlow.save("./models/fig3", bf)
    # new_bf = BitFlow.load("./models/fig3")

    # new_bf.train(epochs=5)

    # assert new_bf.range_lr == bf.range_lr

    return


# def test_fig3_4():

#     dag = gen_fig3()

#     bf = BitFlow(dag, {"z": 4.}, {'a': (-3., 2.),
#                                   'b': (4., 8.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=50000, testing_size=10000, distribution=2, incorporate_ulp_loss=False, batch_size=16, test_optimizer=True, graph_loss=False)
#     bf.train(epochs=10, decay=0.8)

#     # # check saving object works
#     # BitFlow.save("./models/fig3", bf)
#     # new_bf = BitFlow.load("./models/fig3")

#     # new_bf.train(epochs=5)

#     # assert new_bf.range_lr == bf.range_lr

#     return

# def test_fig3_12():

#     dag = gen_fig3()

#     bf = BitFlow(dag, {"z": 12.}, {'a': (-3., 2.),
#                                    'b': (4., 8.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=50000, testing_size=10000, distribution=2, incorporate_ulp_loss=False, batch_size=4, test_optimizer=True, graph_loss=False)
#     bf.train(epochs=2, limit=0.00025)

#     # # check saving object works
#     # BitFlow.save("./models/fig3", bf)
#     # new_bf = BitFlow.load("./models/fig3")

#     # new_bf.train(epochs=5)

#     # assert new_bf.range_lr == bf.range_lr

#     return


# def test_ex1():
#     t0 = time.time()

#     dag = gen_ex1()

#     bf = BitFlow(dag, {"z_1": 0., "z_2": 0.}, {
#         'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=10000, testing_size=2000, incorporate_ulp_loss=False, test_optimizer=True)
#     bf.train(epochs=20)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")
#     return


# def test_ex2():
#     t0 = time.time()

#     dag = gen_ex2()

#     bf = BitFlow(dag, {"res": 8.}, {
#         'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=10000, testing_size=2000, incorporate_ulp_loss=False, test_optimizer=True)
#     bf.train(epochs=5)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")
#     return


# def test_ex3():
#     t0 = time.time()

#     dag = gen_ex3()

#     bf = BitFlow(dag, {"res": 8.}, {
#         'a': (-5, 5), 'b': (-5, 5), 'c': (-5, 5), 'd': (-5, 5)}, lr=1e-2, range_lr=1e-2, train_range=True, training_size=10000, testing_size=4000, incorporate_ulp_loss=False, test_optimizer=True, distribution=2)
#     bf.train(epochs=15)

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
#     print("\n=== RGB ===")

#     dag = RGB_to_YCbCr()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"col_1": 8., "col_2": 8., "col_3": 8.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=10, limit=-1e-3, decay=0.6)

#     rounded_dag = bf.rounded_dag
#     original_dag = bf.original_dag

#     verilog_gen = BitFlow2Verilog(
#         "rgb_to_ycbcr", bf.P, bf.R, bf.filtered_vars, original_dag, {"col_1": 8., "col_2": 8., "col_3": 8.})
#     verilog_gen.evaluate()

#     # Sample Matrix Product
#     test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return


# def test_rgb_case_study_4():
#     print("\n=== RGB ===")

#     dag = RGB_to_YCbCr()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"col_1": 4., "col_2": 4., "col_3": 4.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=10, limit=-5e-2, decay=0.6)

#     # Sample Matrix Product
#     test = {"r": 252., "g": 59., "b": 32., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_rgb_case_study_12():
#     print("\n=== RGB ===")

#     dag = RGB_to_YCbCr()

#     params = dict(
#         training_size=200000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"col_1": 12., "col_2": 12., "col_3": 12.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=5, limit=0.00001, decay=0.0)

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
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=0,
#         custom_data=(train_gen, test_gen),
#         test_optimizer=True
#     )

#     bf = BitFlow(dag, {"col_1": 8., "col_2": 8., "col_3": 8.}, {
#         'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
#     bf.train(epochs=10)

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

    p0 = Mul(Add(a00, a11), Add(b00, b11), name="p0")
    p1 = Mul(Add(a10, a11), b00, name="p1")
    p2 = Mul(a00, Sub(b01, b11), name="p2")
    p3 = Mul(a11, Sub(b10, b00), name="p3")
    p4 = Mul(b11, Add(a00, a01), name="p4")
    p5 = Mul(Add(b00, b01), Sub(a10, a00), name="p5")
    p6 = Mul(Add(b10, b11), Sub(a01, a11), name="p6")

    y00 = Add(Sub(Add(p0, p3), p4), p6, name="y00")
    y01 = Add(p2, p4, name="y01")
    y10 = Add(p1, p3, name="y10")
    y11 = Add(Sub(Add(p0, p2), p1), p5, name="y11")

    matrix_dag = Dag(outputs=[y00, y01, y10, y11], inputs=[
        a00, a01, a10, a11, b00, b01, b10, b11])

    return matrix_dag


# def test_matrix_case_study():
#     print("\n=== MATRIX ===")

#     dag = matrix_multiply()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=True
#     )

#     bf = BitFlow(dag, {"y00": 8., "y01": 8., "y10": 8., "y11": 8.}, {
#         'a00': (-5., 5.), 'a01': (-5., 5.), 'a10': (-5., 5.), 'a11': (-5., 5.), 'b00': (-5., 5.), 'b01': (-5., 5.), 'b10': (-5., 5.), 'b11': (-5., 5.)}, **params)
#     bf.train(epochs=15, limit=-1e-3, decay=0.8)

#     # Sample Matrix Product (Identity Product)
#     test = {"a00": 1., "a01": 1., "a10": 1., "a11": 1., "b00": 1.,
#             "b01": 0., "b10": 0., "b11": 1., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_matrix_case_study_4():
#     print("\n=== MATRIX ===")

#     dag = matrix_multiply()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"y00": 4., "y01": 4., "y10": 4., "y11": 4.}, {
#         'a00': (-5., 5.), 'a01': (-5., 5.), 'a10': (-5., 5.), 'a11': (-5., 5.), 'b00': (-5., 5.), 'b01': (-5., 5.), 'b10': (-5., 5.), 'b11': (-5., 5.)}, **params)
#     bf.train(epochs=10, limit=-1e-2, decay=0.5)

#     # Sample Matrix Product (Identity Product)
#     test = {"a00": 1., "a01": 1., "a10": 1., "a11": 1., "b00": 1.,
#             "b01": 0., "b10": 0., "b11": 1., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_matrix_case_study_12():
#     print("\n=== MATRIX ===")

#     dag = matrix_multiply()

#     params = dict(
#         training_size=100000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=True,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"y00": 12., "y01": 12., "y10": 12., "y11": 12.}, {
#         'a00': (-5., 5.), 'a01': (-5., 5.), 'a10': (-5., 5.), 'a11': (-5., 5.), 'b00': (-5., 5.), 'b01': (-5., 5.), 'b10': (-5., 5.), 'b11': (-5., 5.)}, **params)
#     bf.train(epochs=5, limit=0.00002, decay=0.0)

#     # Sample Matrix Product (Identity Product)
#     test = {"a00": 1., "a01": 1., "a10": 1., "a11": 1., "b00": 1.,
#             "b01": 0., "b10": 0., "b11": 1., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return


def generate_poly_approx(a, b, c, d):
    x = Input(name="x")
    a = Constant(a, name="a")
    b = Constant(b, name="b")
    c = Constant(c, name="c")
    d = Constant(d, name="d")

    output = Mul(
        Add(Mul(Add(Mul(Add(Mul(a, x), b), x), c), x), d), x, name="res")

    approx_dag = Dag(outputs=[output], inputs=[x])

    return approx_dag


# def test_poly_approx():
#     print("\n=== POLY APPROX ===")
#     t0 = time.time()

#     dag = generate_poly_approx(-0.25, 0.333, -0.5, 1.)

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=True
#     )

#     bf = BitFlow(dag, {"res": 8.}, {"x": (0., 1.5)}, **params)
#     bf.train(epochs=10, limit=-1e-3, decay=0.8)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     test = {"x": 0.3, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_poly_approx_4():
#     print("\n=== POLY APPROX ===")
#     t0 = time.time()

#     dag = generate_poly_approx(-0.25, 0.333, -0.5, 1.)

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"res": 4.}, {"x": (0., 1.5)}, **params)
#     bf.train(epochs=10, limit=-1e-3, decay=0.8)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     test = {"x": 0.3, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_poly_approx_12():
#     print("\n=== POLY APPROX ===")
#     t0 = time.time()

#     dag = generate_poly_approx(-0.25, 0.333, -0.5, 1.)

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False,
#         graph_loss=False
#     )

#     bf = BitFlow(dag, {"res": 12.}, {"x": (0., 1.5)}, **params)
#     bf.train(epochs=2, limit=-5e-4)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     test = {"x": 0.3, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return


def generate_basic_lut():
    x = Input(name="x")
    amplitude = Input(name="a")
    shift = Input(name="b")

    output = Add(Mul(amplitude, LookupTable(
        np.sin, x)), shift, name="res")
    # output = Add(Mul(amplitude, x), shift, name="res")

    sin_dag = Dag(outputs=[output], inputs=[x, amplitude, shift])

    return sin_dag


# def test_basic_lut():
#     t0 = time.time()

#     dag = generate_basic_lut()

#     params = dict(
#         training_size=10000,
#         testing_size=2000,
#         batch_size=16,
#         lr=1e-3,
#         train_range=True,
#         range_lr=1e-3,
#         distribution=0,
#         test_optimizer=True,
#         incorporate_ulp_loss=False
#     )

#     bf = BitFlow(dag, {"res": 8.}, {"x": (-2., 2.),
#                                     "a": (1., 5.), "b": (-3, 3)}, **params)
#     bf.train(epochs=20)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     test = {"x": 0.3, "a": 2., "b": 1., "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return


def generate_square_wave():
    x = Input(name="x")

    a = LookupTable(np.sin, x)
    b = Add(Mul(Constant(1/3.), LookupTable(np.sin, Mul(Constant(3.), x))), a)
    c = Add(Mul(Constant(1/5.), LookupTable(np.sin, Mul(Constant(5.), x))), b)
    d = Add(Mul(Constant(1/7.), LookupTable(np.sin, Mul(Constant(7.), x))), c)
    e = Add(Mul(Constant(1/9.), LookupTable(np.sin,
                                            Mul(Constant(9.), x))), d, name="res")

    dag = Dag(outputs=[e], inputs=[x])

    return dag


# def test_square_wave():
#     print("=== SQUARE WAVE ===")

#     dag = generate_square_wave()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False
#     )

#     bf = BitFlow(dag, {"res": 8.}, {"x": (-3., 3.)}, **params)
#     bf.train(epochs=10, decay=0.6, limit=-1e-3)

#     test = {"x": 2.1, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def test_square_wave_4():
#     print("=== SQUARE WAVE ===")

#     dag = generate_square_wave()

#     params = dict(
#         training_size=50000,
#         testing_size=10000,
#         batch_size=16,
#         lr=1e-2,
#         train_range=True,
#         range_lr=1e-2,
#         distribution=2,
#         test_optimizer=True,
#         incorporate_ulp_loss=False
#     )

#     bf = BitFlow(dag, {"res": 4.}, {"x": (-3., 3.)}, **params)
#     bf.train(epochs=10)

#     test = {"x": 2.1, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return

# def generate_vector_example():
#     X = Input(name="X")

#     out = Reduce(Concat(Select(Select(X, 0), 0), Select(
#         Select(X, 1), 0), concat_dim=0), reduce_dim=0, name="res")

#     dag = Dag(outputs=[out], inputs=[X])

#     return dag


# def test_vector_ex():
#     dag = generate_vector_example()

#     params = dict(
#         training_size=10000,
#         testing_size=2000,
#         batch_size=16,
#         lr=1e-3,
#         train_range=True,
#         range_lr=1e-3,
#         distribution=0,
#         test_optimizer=False,
#         incorporate_ulp_loss=True
#     )

#     bf = BitFlow(dag, {"res": 4.}, {"X": [(0., 1.), (0., 1.)]}, **params)
#     bf.train(epochs=10)

#     return

def generate_basic(a):
    x = Input(name="x")
    a = Constant(a, name="a")

    output = Mul(a, x, name="res")

    approx_dag = Dag(outputs=[output], inputs=[x])

    return approx_dag


# def test_basic():
#     print("\n=== BASIC ===")
#     t0 = time.time()

#     dag = generate_basic(-0.25)

#     params = dict(
#         training_size=10000,
#         testing_size=2000,
#         batch_size=16,
#         lr=4e-3,
#         train_range=True,
#         range_lr=4e-3,
#         distribution=0,
#         test_optimizer=True,
#         incorporate_ulp_loss=True,
#         graph_loss=True
#     )

#     bf = BitFlow(dag, {"res": 8.}, {"x": (0., 1.)}, **params)
#     bf.train(epochs=5)

#     print(f"TIME: {time.time() - t0} SECONDS ELAPSED")

#     test = {"x": 0.3, "P": bf.P, "R": bf.R, "O": bf.O}
#     print(bf.model(**test))

#     return
