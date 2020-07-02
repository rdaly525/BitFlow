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

        X = [[torch.tensor([3]), torch.tensor([4])],
             [torch.tensor([3]), torch.tensor([4])]]

        Y = [torch.tensor([3]), torch.tensor([4])]

        W = [torch.tensor([12.], requires_grad=True), torch.tensor([12.], requires_grad=True), torch.tensor([12.], requires_grad=True)]

        # 5. Loss function
        def compute_loss(y, target):
            #L2 norm squared
            return torch.sqrt(target - y)

        # 6. Run torch on DAG
        epochs = 3
        lr_rate = 0.001
        opt = torch.optim.SGD(W, lr=.001)
        losslog=[]
        for e in range(epochs):
            for i in range(2):
                data = {"X": X[i], "W": W}
                y = model(**data)
                print(y)

                loss = compute_loss(y, Y[i], W)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if t%10==0:
                    print(loss)


        # print(evaluator)
