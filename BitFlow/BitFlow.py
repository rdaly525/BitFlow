from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from .Eval.TorchEval import TorchEval
from .Optimization import BitFlowVisitor

import torch
from torch.utils import data

class BitFlow:
    def gen_model(self, dag):
        self.evaluator = TorchEval(dag)
        def model(**kwargs):
            return self.evaluator.eval(**kwargs)
        return model

    def gen_data(self, model, output_precision, num, size_x, size_w, mean, std):
        X = []
        Y = []
        W = torch.Tensor(1, size_w).fill_(64)[0]
        for i in range(num):
            new_x = torch.normal(mean, std, (1,size_x))
            inputs = {"X": new_x[0], "W": W, "O": output_precision}
            new_y = model(**inputs)

            X.append(new_x.tolist()[0])
            Y.append(new_y.tolist()[0])

        return torch.tensor(X), torch.tensor(Y)

    def update_dag(self, dag):
        # 1. Define the DAG and weight nodes
        W = Input(name="W")
        O = Input(name="O")
        X = Input(name="X")

        a = Round(Select(X, 0, name="a"), W[0])
        b = Round(Select(X, 1, name="b"), W[1])
        c = Round(Constant(4, name="c"), W[2])

        # 2. Add Round Nodes after +/-/* in DAG
        d = Round(Mul(a, b, name="d"), W[3])
        e = Round(Add(d, c, name="e"), W[4])
        z = Round(Sub(e, b, name="z"), O[0])

        fig3 = Dag(outputs=[z], inputs=[X,W,O])

        return fig3

    def __init__(self, dag, precision, mean=5., std=3.):
        evaluator = NumEval(dag)
        evaluator.eval(a=1, b=1)

        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name

        filtered_vars = []
        for var in vars:
            if var != 'z':
                filtered_vars.append(var)

        error_fn = visitor.errors['z'].getExecutableError()
        area_fn = visitor.area_fn

        error_fn = f"2**(-{precision}-1) - (" + error_fn + ")"

        exec(f'''def AreaOptimizerFn(W):
             {','.join(filtered_vars)} = W
             return  {area_fn}''', globals())

        exec(f'''def ErrorConstraintFn(x):
             {','.join(filtered_vars)} = x
             return  {error_fn}''', globals())

        print(error_fn)
        print(area_fn)
        print(filtered_vars)

        dag = self.update_dag(dag)
        model = self.gen_model(dag)

        database_size = 100
        input_size = 2
        weight_size = 6

        # X = torch.tensor([[3., 4.], [-3., 2.], [1., 8.]])
        # Y = torch.tensor([12., -4., 4.])

        O = torch.tensor([8.])
        X, Y = self.gen_data(model, O, database_size, input_size, weight_size, mean, std)

        W = torch.tensor([12., 12., 12., 12., 12.], requires_grad=True)

        # Loss function
        def compute_loss(target, y, W, iter):
            area = torch.tensor(AreaOptimizerFn(W.tolist()))
            error = torch.tensor(ErrorConstraintFn(W.tolist())) # >= 0
            L1 = target - y

            loss = L1 + area + max(-error, 0)

            if iter % 500 == 0:
                print(f"AREA: {area}")
                print(f"ERROR: {error}")
                print(f"LOSS: {loss[0]}")

            return loss

        # Run torch on DAG
        epochs = 500
        lr_rate = 0.05

        opt = torch.optim.AdamW([W], lr=lr_rate)

        iter = 0
        for e in range(epochs):
            for i in range(database_size):
                inputs = {"X": X[i], "W": W, "O": O}
                y = model(**inputs)
                loss = compute_loss(Y[i], y, W, iter)
                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1

        test = {"X": torch.tensor([4., 4.]), "W": W, "O": O}
        print(torch.ceil(W))
        print(model(**test))
