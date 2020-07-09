from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from .Eval.TorchEval import TorchEval
from .Optimization import BitFlowVisitor, BitFlowOptimizer

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

    def gen_data(self, model, output_precision, num, size_x, size_w, data_range):
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
            new_x = (data_range[1]-data_range[0]) * \
                torch.rand((1, size_x)) + data_range[0]
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

    def round_to_precision(self, num, precision):
        scale = 2.0**precision
        return torch.round(num * scale) / scale

    def is_within_ulp(self, num, truth, precision):
        return (self.round_to_precision(truth, precision) + 2**-(precision + 1) > num and self.round_to_precision(truth, precision) - 2**-(precision + 1) < num)

    def calc_accuracy(self, name, X, Y, W, O, precision, testing_size, model, should_print):
        success = 0
        print(f"\n##### {name} SET ######")
        for i in range(testing_size):
            sample_X = X[i]
            sample_Y = Y[i]
            inputs = {"X": sample_X, "W": W, "O": O}
            res = model(**inputs)
            if (self.is_within_ulp(res, sample_Y, precision)):
                success += 1
            else:
                if should_print:
                    print(f"prediction: {res[0]} : true: {sample_Y}")
        acc = (success * 1.)/testing_size
        print(f"accuracy: {acc}")

    def __init__(self, dag, precision, data_range=(-2., 2.)):

        # Run a basic evaluator on the DAG to construct error and area functions
        evaluator = NumEval(dag)

        # TODO: generalize this line
        evaluator.eval(a=3, b=8)

        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        bfo = BitFlowOptimizer(evaluator, 'z', precision)
        bfo.calculateInitialValues()

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
        training_size = 2000
        testing_size = 200
        input_size = 2  # TODO: adapt to DAG
        weight_size = 5  # TODO: adapt to DAG
        epochs = 5
        lr_rate = 0.01

        # output without grad TODO: generalize to DAG
        O = torch.tensor([precision])

        # generate testing/training data
        train_X, train_Y = self.gen_data(model, O, training_size,
                                         input_size, weight_size, data_range)
        test_X, test_Y = self.gen_data(model, O, testing_size,
                                       input_size, weight_size, data_range)

        # weights matrix
        W = torch.Tensor(1, weight_size).fill_(bfo.initial)[0]
        init_W = W.clone()
        print(W)
        W.requires_grad = True

        # Loss function
        def compute_loss(target, y, W, iter):
            area = torch.tensor(AreaOptimizerFn(W.tolist()))

            N = 1000
            M = 10000
            L = 10000

            # >= 0
            ulp_err = K * (int(self.is_within_ulp(y, target, precision)
                               [0]) - 1)

            constraint_err = torch.max(-torch.tensor(
                ErrorConstraintFn(W.tolist())), torch.zeros(1))  # >= 0

            L2 = torch.sum((y-target)**2)

            loss = (L * L2 + M * constraint_err + area/K) ** 2

            if iter % 500 == 0:
                print(
                    f"iteration {iter} of {epochs * training_size} ({(iter * 100.)/(epochs * training_size)}%)")
                print(f"AREA: {area}")
                print(f"ERROR: {constraint_err[0]}")
                print(f"LOSS: {loss[0]}")

            return loss

        # Set up optimizer
        opt = torch.optim.Adam([W], lr=lr_rate)

        # Run training process
        print("\n##### TRAINING ######")
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

        print(W)
        W = torch.ceil(W)

        self.calc_accuracy("TEST", test_X, test_Y, W,
                           O, precision, testing_size, model, True)
        self.calc_accuracy("UFB TEST", test_X, test_Y, init_W,
                           O, precision, testing_size, model, False)

        self.calc_accuracy("TRAIN", train_X, train_Y, W,
                           O, precision, testing_size, model, False)
        self.calc_accuracy("UFB TRAIN", train_X, train_Y, init_W,
                           O, precision, testing_size, model, False)

        print("\n##### SAMPLE ######")

        # Basic sample (truth value = 16)
        test = {"X": torch.tensor([4., 4.]), "W": W, "O": O}
        print(W)
        print(init_W)
        print(model(**test))
        print(self.is_within_ulp(model(**test),
                                 torch.tensor([16.]), precision))
