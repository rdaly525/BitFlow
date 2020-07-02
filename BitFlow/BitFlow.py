from node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode
from IA import Interval
from Eval.IAEval import IAEval
from Eval.NumEval import NumEval
from Eval.TorchEval import TorchEval
import torch
#from torch.functions import Round as tRound
from torch.autograd import Variable

class BitFlow:
    def __init__(self):
        # STEPS
        # 1. Define DAG
        # 2. Add Round Nodes after +/-/* in DAG
        # 3.
        X = [[-3, 4], [2, 8]]
        W = [12, 12, 12]

        # 1. Define the DAG and weight nodes
        w0 = Input(name="w0")
        w1 = Input(name="w1")
        w2 = Input(name="w2")

        a = Input(name="a")
        b = Input(name="b")
        c = Constant(4, name="c")

        # 2. Add Round Nodes after +/-/* in DAG
        d = Round(Mul(a, b, name="d"), w0)
        e = Round(Add(d, c, name="e"), w1)
        z = Round(Sub(e, b, name="z"), w2)

        fig3 = Dag(output=z, inputs=[a,b])
        evaluator = TorchEval(fig3)

        # 3. Evaluate the DAG
        a, b = Variable(torch.Tensor([X[1][0]]), requires_grad=False), Variable(torch.Tensor([X[1][1]]), requires_grad=False)

        weights = [Variable(torch.Tensor([12]), requires_grad=True), Variable(torch.Tensor([12]), requires_grad=True), Variable(torch.Tensor([12]), requires_grad=True)]
        print(evaluator.eval(a=a, b=b, w0=weights[0], w1=weights[1], w2=weights[2]))

        # 4. Set up learning
        epochs = 5
        lr_rate = 0.001

        # 5. Loss function
        def compute_loss(y, target, W):
            #L2 norm squared
            #Ross's solution
            target_loss = 1*torch.sum((y-target)**2)
            w_loss = 1000*torch.sum(torch.max(-W + 0.5, torch.zeros(2)))
            loss = target_loss + w_loss
            return loss

        # 6. Run torch on DAG
        epochs = 3
        opt = torch.optim.SGD([W], lr=.001)
        losslog=[]
        for e in range(epochs):
            for t, (X, target_y) in enumerate(train_gen):
                y = model(X, W)
                loss = compute_loss(y, target_y, W)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if t%10==0:
                    print(loss)





        print(evaluator)



x = BitFlow()
