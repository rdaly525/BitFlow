from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from DagVisitor import Visitor
from BitFlow.IA import Interval
from BitFlow.Eval import IAEval, NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.BitFlow import BitFlow
import torch
from BitFlow.utils import Imgs2Dataset

from torch.utils import data


def test_image2data():
    converter = Imgs2Dataset()
    dataset, size = converter.convert_rgb2ycbcr('./data/imgs/')
    train_set, test_set = torch.utils.data.random_split(
        dataset, [int(round(0.8 * size)), int(round(0.2 * size))])

    return


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

    dag = gen_fig3()

    bf = BitFlow(dag, {"z": 8.}, {'a': (-3., 2.),
                                  'b': (4., 8.)}, lr=8e-4, range_lr=5e-4, train_range=True, training_size=1000, testing_size=200, distribution=0)
    bf.train(epochs=1)

    # check saving object works
    BitFlow.save("./models/fig3", bf)
    new_bf = BitFlow.load("./models/fig3")
    new_bf.reset()

    new_bf.train(epochs=5)

    assert new_bf.range_lr == bf.range_lr

    return


def test_ex1():
    dag = gen_ex1()

    bf = BitFlow(dag, {"z_1": 5., "z_2": 8.}, {
        'a': (-3., 2.), 'b': (4., 8.), 'c': (-1., 1.)}, train_range=True)
    bf.train(epochs=10)
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
    dag = RGB_to_YCbCr()

    params = dict(
        training_size=200,
        testing_size=20,
        batch_size=16,
        lr=1e-4,
        train_range=True,
        range_lr=1e-4,
        distribution=0
    )

    bf = BitFlow(dag, {"col_1": 10., "col_2": 10., "col_3": 10.}, {
        'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
    bf.train(epochs=1)

    # Sample Matrix Product
    test = {"r": 2., "g": 4., "b": -3., "P": bf.P, "R": bf.R, "O": bf.O}
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
        lr=5e-4,
        train_range=True,
        range_lr=1e-4,
        distribution=0,
        custom_data=(train_gen, test_gen),
        test_optimizer=False
    )

    bf = BitFlow(dag, {"col_1": 0., "col_2": 0., "col_3": 0.}, {
        'r': (0., 255.), 'b': (0., 255.), 'g': (0., 255.)}, **params)
    bf.train(epochs=1)

    # Sample Matrix Product
    test = {"r": 125., "g": 125., "b": 125., "P": bf.P, "R": bf.R, "O": bf.O}
    print(bf.model(**test))

    return
