from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from .Eval.TorchEval import TorchEval
from .Optimization import BitFlowVisitor, BitFlowOptimizer

import torch
from torch.utils import data

import random
import math


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

    def custom_round(self, W, factor=0.5):
        W = W.tolist()
        for (index, weight) in enumerate(W):
            f, _ = math.modf(weight)
            if f < factor:
                W[index] = math.floor(weight)
            else:
                W[index] = math.ceil(weight)
        return torch.tensor(W)

    def gen_data(self, model, dataset_size, size_w, size_output, data_range, true_width=20., dist=0):
        """ Generates ground-truth data from user specifications and model.

        Args:
            model: A dag already set up for torch evaluatiion
            output_precision: The requested number of precision bits on the output node
            num: Number of samples to generate
            size_w: The size of the weight array
            dist: Type of distribution to use
                0 ==> UNIFORM
                1 ==> NORMAL
            mean, std: statistics for normal distribution to generate data from

        Returns:
            (X, Y): generated data
        """

        class Dataset(data.Dataset):
            def __init__(self, model, dataset_size, size_w, size_output, data_range, true_width, dist):
                self.X = {k: [] for k in data_range.keys()}
                self.Y = []

                # TODO: create a mode to set constant precision for inputs

                W = torch.Tensor(1, size_w).fill_(true_width)[0]
                random.seed(42)

                for i in range(dataset_size):
                    for key in data_range:
                        input_range = data_range[key]
                        if dist == 1:
                            mean = (input_range[1]-input_range[0])/2
                            std = (mean - input_range[0])/2
                            self.X[key].append(torch.tensor(
                                random.normalvariate(mean, std)))
                        else:
                            self.X[key].append(torch.tensor(
                                random.uniform(*input_range)))

                    inputs = {k: self.X[k][i] for k in data_range.keys()}
                    inputs["W"] = W
                    inputs["O"] = torch.Tensor(
                        1, size_output).fill_(true_width)[0]
                    print(inputs)
                    new_y = model(**inputs)
                    print(new_y)
                    assert 0
                    self.Y.append(new_y.tolist()[0])

                print(self.X)
                print(self.Y)

            def __len__(self):
                return len(self.X[list(data_range.keys())[0]])

            def __getitem__(self, index):
                return {k: self.X[k][index] for k in data_range.keys()}, self.Y[index]

        return Dataset(model, dataset_size, size_w, size_output, data_range, true_width, dist)

    def update_dag(self, dag):
        """ TODO: Adds Round Nodes, Weight Input Nodes (grad allowed) and Output Precision Nodes (grad not allowed)

        Args:
            dag: Input dag

        Returns:
            updated dag: A dag which BitFlow can be run on
        """
        W = Input(name="W")
        O = Input(name="O")

        a_input = Input(name="a")
        b_input = Input(name="b")

        a = Round(a_input, W[0])
        b = Round(b_input, W[1])
        c = Round(Constant(4.3, name="c"), W[2])

        d = Round(Mul(a, b, name="d"), W[3])
        e = Round(Add(d, c, name="e"), W[4])
        z = Round(Sub(e, b, name="z"), O[0])

        fig3 = Dag(outputs=[z], inputs=[a_input, b_input, W, O])

        return fig3

    def round_to_precision(self, num, precision):
        scale = 2.0**precision
        return torch.round(num * scale) / scale

    def is_within_ulp(self, num, truth, precision):
        return (self.round_to_precision(truth, precision) + 2**-(precision + 1) > num and self.round_to_precision(truth, precision) - 2**-(precision + 1) < num)

    def calc_accuracy(self, name, test_gen, W, O, precision, model, should_print):
        success = 0
        total = 0
        print(f"\n##### {name} SET ######")
        for t, (X, Y) in enumerate(test_gen):
            for i in range(Y.shape[0]):
                sample_X = X[i]
                sample_Y = Y[i]

                # FIX!!!
                inputs = {"X": sample_X, "W": W, "O": O}
                res = model(**inputs)
                total += 1
                if (self.is_within_ulp(res, sample_Y, precision)):
                    success += 1
                else:
                    if should_print:
                        print(
                            f"prediction: {res[0]} : true: {self.round_to_precision(sample_Y, precision)}")
        acc = (success * 1.)/total
        print(f"accuracy: {acc}")

    def within_ulp_err(self, num, truth, precision):
        return torch.abs(truth - num) - 2 ** -(precision + 1)

    def hasNegatives(self, tensor):
        vals = tensor < 0
        return vals.any()

    def __init__(self, dag, precision, data_range={'a': (-3., 2.), 'b': (4., 8.)}):
        # TODO: construct data_range from dag.inputs

        # Run a basic evaluator on the DAG to construct error and area functions
        evaluator = NumEval(dag)

        eval_dict = data_range.copy()
        for key in eval_dict:
            eval_dict[key] = int(max(abs(eval_dict[key][0]),
                                     abs(eval_dict[key][1])))

        evaluator.eval(**eval_dict)

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
        training_size = 1000
        testing_size = 200
        input_size = 2  # TODO: adapt to DAG
        weight_size = 5  # TODO: adapt to DAG
        output_size = 1  # TODO: adapt to DAG
        epochs = 50
        batch_size = 8

        # lr -> (1e-7 (2 bits), 5e-6 (8 bits))
        lr_rate = 8e-4

        # output without grad TODO: generalize to DAG
        O = torch.Tensor(
            1, output_size).fill_(precision)[0]

        params = dict(
            batch_size=batch_size
        )

        # generate testing/training data
        training_set = self.gen_data(
            model, training_size, weight_size, output_size, data_range)
        train_gen = data.DataLoader(training_set, **params)
        test_set = self.gen_data(
            model, testing_size, weight_size, output_size, data_range)
        test_gen = data.DataLoader(test_set, **params)

        # weights matrix
        W = torch.Tensor(weight_size).fill_(bfo.initial)
        init_W = W.clone()
        print(W)
        W += 2
        W.requires_grad = True
        init_W.requires_grad = True

        self.R = 1e20
        self.initR = self.R
        self.prevR = self.R

        # Loss function
        def compute_loss(target, y, W, iter, error_type=1, should_print=True, shouldErrorCheck=False):
            """
            Args:
                error_type:
                    1 ==> Paper Error
                    2 ==> Soft-Loss on ULP

            """
            area = torch.tensor(AreaOptimizerFn(W.tolist()))

            loss = 0
            if error_type == 1:

                # Calculate erros
                constraint_err = torch.tensor(
                    ErrorConstraintFn(W.tolist()))
                ulp_error = torch.mean(torch.sum(
                    self.within_ulp_err(torch.tensor(y), target, precision)))

                # Sanity error check
                if shouldErrorCheck:
                    if self.hasNegatives(constraint_err):
                        raise ValueError(
                            f"ERR NEGATIVE: {constraint_err}, {W}, {area}")

                # If ulp error is reasonable, relax error constraints
                decay = 0.95
                if torch.abs(ulp_error) < 1:
                    self.prevR = self.R
                    self.R *= decay
                else:
                    self.R = self.initR

                L2 = torch.sum((y-target)**2)

                S = 1
                Q = 100
                loss = (Q * L2 + self.R *
                        torch.exp(-10 * constraint_err) + S * area)/batch_size

            else:
                ulp_error = precision * torch.mean(torch.sum(
                    self.within_ulp_err(torch.tensor(y), target, precision)))

                constraint_err = torch.max(
                    ulp_error, torch.zeros(1))

                constraint_W = 100 * \
                    torch.sum(torch.max(-(W) + 0.5, torch.zeros(len(W))))

                loss = (constraint_err + constraint_W + area/100)/batch_size

            # Catch negative values for area and weights
            if shouldErrorCheck:
                if self.hasNegatives(area):
                    raise ValueError(f"AREA ERR: {W}, {area}")

                if self.hasNegatives(W):
                    raise ValueError(f"WEIGHT ERR: {W}, {area}")

           # Print out model details every so often
            if iter % 1000 == 0 and should_print == True:
                print(
                    f"iteration {iter} of {epochs * training_size/batch_size} ({(iter * 100.)/(epochs * training_size/batch_size)}%)")
                print(f"AREA: {area}")
                print(f"ERROR: {constraint_err}")
                print(f"WEIGHTS: {W}")
                if error_type == 1:
                    print(f"ERROR CONST: {self.R}")
                print(f"LOSS: {loss}")

            return loss

        # Set up optimizer
        opt = torch.optim.AdamW([W], lr=lr_rate)

        # Run training process
        print("\n##### TRAINING ######")
        iter = 0
        for e in range(epochs):
            for t, (inputs, target_y) in enumerate(train_gen):
                inputs["W"] = W
                inputs["O"] = O

                print(f"INPUTS: {inputs}")

                y = model(**inputs)

                print(f"PREDS: {y}")
                print(f"TRUE: {target_y}")

                loss = compute_loss(target_y, y, W, iter,
                                    error_type=1, should_print=True)

                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1

        print(W)
        print(iter)
        W = self.custom_round(W, factor=0.2)

        self.calc_accuracy("TEST", test_gen, W,
                           O, precision, model, True)
        self.calc_accuracy("UFB TEST", test_gen, init_W,
                           O, precision, model, False)

        self.calc_accuracy("TRAIN", train_gen, W,
                           O, precision, model, False)
        self.calc_accuracy("UFB TRAIN", train_gen, init_W,
                           O, precision, model, False)

        print("\n##### MODEL DETAILS #####")
        print(f"ERROR: {ErrorConstraintFn(W.tolist())}")
        print(f"AREA: {AreaOptimizerFn(W.tolist())}")

        print("\n##### SAMPLE ######")

        # Basic sample (truth value = 16)
        test = {"X": torch.tensor([2., 4.]), "W": W, "O": O}
        print(W)
        print(init_W)
        print(model(**test))
        print(self.is_within_ulp(model(**test),
                                 torch.tensor([8.3]), precision))

        print("\n##### FROM OPTIMIZER ######")
        bfo.solve()
        test = [bfo.fb_sols['a'], bfo.fb_sols['b'],
                bfo.fb_sols['c'], bfo.fb_sols['d'], bfo.fb_sols['e']]
        print(f"ERROR: {ErrorConstraintFn(test)}")
        print(f"AREA: {AreaOptimizerFn(test)}")

        self.calc_accuracy("OPTIMIZER TEST", test_gen, torch.tensor(test),
                           O, precision, model, False)
