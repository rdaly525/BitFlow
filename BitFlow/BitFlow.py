from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from .Eval.TorchEval import TorchEval

import torch
from torch.utils import data

class BitFlow:
    def gen_model(self, dag):
        evaluator = TorchEval(dag)
        def model(**kwargs):
            return evaluator.eval(**kwargs)
        return model

    def __init__(self, dag):

        model = self.gen_model(dag)
        database_size = 2

        X = torch.tensor([[3., 4.], [-3., 2.], [1., 8.]], requires_grad=False)
        Y = torch.tensor([12., -4., 4.], requires_grad=False)

        W = torch.tensor([12., 12., 12.], requires_grad=True)

        # Loss function
        def compute_loss(target, y):
            #L1 norm
            return target-y

        # Run torch on DAG
        epochs = 64
        lr_rate = 0.001

        opt = torch.optim.SGD([W], lr=lr_rate)

        for e in range(epochs):
            for i in range(database_size):
                inputs = {"X": X[i], "W": W}
                y = model(**inputs)

                loss = compute_loss(Y[i], y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        test = {"X": torch.tensor([4., 4.]), "W": W}
        print(W)
        print(model(**test))
