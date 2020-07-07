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
        """ Sets up a given dag for torch evaluation.

        Args: Input dag (should already have Round nodes)

        Returns: A trainable model
        """
        self.evaluator = TorchEval(dag)

        def model(**kwargs):
            return self.evaluator.eval(**kwargs)
        return model

    def gen_data(self, model, output_precision, num, size_x, size_w, mean, std):
        """ Generates ground-truth data from user specifications and model.

        Args:
            model: A dag already set up for torch evaluatiion
            TODO: generalize to multiple outputs
            output_precision: The requested number of precision bits on the output node
            num: Number of samples to generate
            size_x : The size of the input array
            size_w: The size of the weight array
            mean, std: statistics for normal distribution to generate data from

        Returns:
            (X, Y): generated data
        """
        X = []
        Y = []
        W = torch.Tensor(1, size_w).fill_(64)[0]
        for i in range(num):
            new_x = torch.normal(mean, std, (1, size_x))
            inputs = {"X": new_x[0], "W": W, "O": output_precision}
            new_y = model(**inputs)

            X.append(new_x.tolist()[0])
            Y.append(new_y.tolist()[0])

        return torch.tensor(X), torch.tensor(Y)

    def update_dag(self, dag):
        """ TODO: Adds Round Nodes, Weight Input Nodes (grad allowed) and Output Precision Nodes (grad not allowed)

        Args:
            dag: Input dag

        Returns:
            updated dag: A dag which BitFlow can be run on
        """
        W = Input(name="W")
        O = Input(name="O")
        X = Input(name="X")

        a = Round(Select(X, 0, name="a"), W[0])
        b = Round(Select(X, 1, name="b"), W[1])
        c = Round(Constant(4, name="c"), W[2])

        d = Round(Mul(a, b, name="d"), W[3])
        e = Round(Add(d, c, name="e"), W[4])
        z = Round(Sub(e, b, name="z"), O[0])

        fig3 = Dag(outputs=[z], inputs=[X, W, O])

        return fig3

    def __init__(self, dag, precision, mean=5., std=3.):

        # Run a basic evaluator on the DAG to construct error and area functions
        evaluator = NumEval(dag)
        evaluator.eval(a=1, b=1)
        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        # Remove output variables from DAG list (these will be our weights)
        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        filtered_vars = []
        for var in vars:
            if var != 'z':
                filtered_vars.append(var)

        # Generate error and area functions from the visitor
        error_fn = visitor.errors['z'].getExecutableError()
        area_fn = visitor.area_fn

        error_fn = f"2**(-{precision}-1) - (" + error_fn + ")"

        exec(f'''def AreaOptimizerFn(W):
             {','.join(filtered_vars)} = W
             return  {area_fn}''', globals())

        exec(f'''def ErrorConstraintFn(x):
             {','.join(filtered_vars)} = x
             return  {error_fn}''', globals())

        # Update the dag with round nodes and set up the model for torch training
        dag = self.update_dag(dag)
        model = self.gen_model(dag)

        # Training details
        training_size = 500
        testing_size = 100
        input_size = 2  # TODO: adapt to DAG
        weight_size = 6  # TODO: adapt to DAG
        epochs = 100
        lr_rate = 0.05

        # output without grad TODO: generalize to DAG
        O = torch.tensor([8.])

        # generate testing/training data
        train_X, train_Y = self.gen_data(model, O, training_size,
                                         input_size, weight_size, mean, std)
        test_X, test_Y = self.gen_data(model, O, testing_size,
                                       input_size, weight_size, mean, std)

        # weights matrix TODO: generalize to DAG
        W = torch.tensor([12., 12., 12., 12., 12.], requires_grad=True)

        # Loss function
        def compute_loss(target, y, W, iter):
            area = torch.tensor(AreaOptimizerFn(W.tolist()))

            # TODO: error is within 1 ulp of truth
            error = torch.tensor(ErrorConstraintFn(W.tolist()))  # >= 0
            L1 = torch.abs(target - y)

            loss = L1 + area + max(-error, 0)

            if iter % 500 == 0:
                print(f"AREA: {area}")
                print(f"ERROR: {error}")
                print(f"LOSS: {loss[0]}")

            return loss

        # Set up optimizer
        opt = torch.optim.AdamW([W], lr=lr_rate)

        # Run training process
        iter = 0
        for e in range(epochs):
            for i in range(training_size):
                inputs = {"X": train_X[i], "W": W, "O": O}
                y = model(**inputs)
                loss = compute_loss(train_Y[i], y, W, iter)
                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1

        # Run testing process
        for i in range(testing_size):
            sample_X = test_X[i]
            sample_Y = test_Y[i]
            # TEST ACCURACY

        # Basic sample (truth value = 16)
        test = {"X": torch.tensor([4., 4.]), "W": W, "O": O}
        print(torch.ceil(W))
        print(model(**test))
