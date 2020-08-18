from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import BitFlowVisitor, BitFlowOptimizer
from BitFlow.AddRoundNodes import AddRoundNodes

import torch
from torch.utils import data

import random
import math

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
# from DagVisitor import Visitor

import torch

from BitFlow.AddRoundNodes import NodePrinter1,BitFlowOptimizer,BitFlowVisitor, AllKeys
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer

import torch.nn as nn
import math

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval

from BitFlow.AddRoundNodes import AddRoundNodes

import torch
from torch.utils import data

import random
import math

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

    def gen_data(self, model, dataset_size, size_w, size_output, data_range, range_bits, true_width=20., dist=0):
        """ Generates ground-truth data from user specifications and model.
        Args:
            model: A dag already set up for torch evaluatiion
            output_precision: The requested number of precision bits on the output node
            num: Number of samples to generate
            size_w: The size of the weight array
            dist: Type of distribution to use
                0 ==> UNIFORM
                1 ==> NORMAL
                2 ==> ARCSINE
            mean, std: statistics for normal distribution to generate data from
        Returns:
            (X, Y): generated data
        """

        class Dataset(data.Dataset):
            def __init__(self, model, dataset_size, size_w, size_output, data_range, range_bits, true_width, dist):
                self.X = {k: [] for k in data_range}
                self.Y = []

                # TODO: create a mode to set constant precision for inputs

                W = torch.Tensor(1, size_w).fill_(true_width)[0]
                torch.manual_seed(42)

                for key in data_range:
                    # Create random tensor
                    input_range = data_range[key]

                    # calculate range bounds using range bits
                    ib = range_bits[key]
                    min_range = -1 * (2 ** (ib - 1))
                    max_range = 2 ** (ib - 1) - 1

                    val = 0
                    if dist == 1:
                        mean = (input_range[1]+input_range[0])/2
                        std = (mean - input_range[0])/3
                        val = torch.normal(
                            mean=mean, std=std, size=(1, dataset_size)).squeeze()
                    elif dist == 2:
                        beta = torch.distributions.beta.Beta(
                            torch.tensor([0.5]), torch.tensor([0.5]))
                        val = (input_range[1] - input_range[0]) * \
                            beta.sample((dataset_size,)).squeeze() + \
                            input_range[0]
                    else:
                        val = (input_range[1] - input_range[0]) * \
                            torch.rand(dataset_size) + input_range[0]

                    val = torch.clamp(val, min_range, max_range)
                    self.X[key] = val

                for i in range(dataset_size):
                    inputs = {k: self.X[k][i] for k in data_range}

                    inputs["W"] = W
                    inputs["O"] = torch.Tensor(
                        1, size_output).fill_(true_width)[0]
                    new_y = model(**inputs)
                    self.Y.append(new_y)

            def __len__(self):
                return len(self.X[list(data_range.keys())[0]])

            def __getitem__(self, index):
                return {k: self.X[k][index] for k in data_range}, self.Y[index]

        return Dataset(model, dataset_size, size_w, size_output, data_range, range_bits, true_width, dist)

    def update_dag(self, dag):
        """
        Args:
            dag: Input dag
        Returns:
            updated dag: A dag which BitFlow can be run on
        """
        W = Input(name="W")
        O = Input(name="O")

        rounder = AddRoundNodes(W, O)
        roundedDag = rounder.doit(dag)

        return roundedDag, rounder.round_count, rounder.input_count, rounder.output_count

    def round_to_precision(self, num, precision):
        if len(precision) > 1:
            num = num.clone()
            scale = 2.0**precision
            for (ind, val) in enumerate(scale):
                num[ind] *= val
            num = torch.round(num)
            for (ind, val) in enumerate(scale):
                num[ind] /= val
            return num
        else:
            scale = 2.0**precision
            return torch.round(num * scale) / scale

    def is_within_ulp(self, num, truth, precision):
        r = torch.abs(num - self.round_to_precision(truth, precision))
        ulp = 2**-(precision + 1)
        if len(precision) > 1:
            sol = torch.ones(r.shape)
            for (y, row) in enumerate(r):
                for (x, val) in enumerate(row):
                    if val > ulp[y]:
                        sol[y][x] = 0
            return sol
        else:
            return(torch.where(r <= ulp, torch.ones(r.shape), torch.zeros(r.shape)))

    def calc_accuracy(self, name, test_gen, W, O, precision, model, should_print):
        success = 0
        total = 0
        print(f"\n##### {name} SET ######")
        for t, (inputs, Y) in enumerate(test_gen):
            inputs["W"] = W
            inputs["O"] = O

            res = model(**inputs)
            if isinstance(res, list):
                res = torch.stack(res)
                Y = torch.stack(Y).squeeze()
            else:
                Y = Y.squeeze()

            ulp = self.is_within_ulp(res, Y, precision)
            success += torch.sum(ulp)
            total += torch.numel(ulp)

            if should_print and len(precision) == 1:
                indices = (ulp == 0).nonzero()[:, 0].tolist()
                for index in indices:
                    print(
                        f"guess: {res[index]}, true: {self.round_to_precision(Y[index], precision)} ")

        acc = (success * 1.)/total
        print(f"accuracy: {acc}")

    def within_ulp_err(self, num, truth, precision):
        diff = torch.abs(truth - num)

        error = 0
        if len(truth.shape) > 1:
            error = torch.unsqueeze(2 ** -(precision + 1), 1)
            error = error.repeat(1, truth.shape[1])
        else:
            error = 2 ** -(precision + 1)

        return torch.abs(diff - error)

    def hasNegatives(self, tensor):
        vals = tensor < 0
        return vals.any()

    def filterDAGFromOutputs(self, visitor, outputs):
        # Remove output variables from DAG list (these will be our weights)
        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        filtered_vars = []
        for var in vars:
            if var not in outputs:
                filtered_vars.append(var)

        return filtered_vars

    def calculateRange(self, evaluator, outputs):
        node_values = evaluator.node_values
        visitor = BitFlowVisitor(node_values)
        visitor.run(evaluator.dag)

        filtered_vars = self.filterDAGFromOutputs(visitor, outputs)

        range_bits = visitor.IBs
        return range_bits, filtered_vars

    def constructOptimizationFunctions(self, dag, outputs, data_range):

        evaluator = NumEval(dag)

        eval_dict = data_range.copy()
        for key in eval_dict:
            print(key)

            eval_dict[key] = 10

        #print(eval_dict)
        #eval_dict = {'X': torch.ones(100,784).fill_(10), 'weight': torch.ones(784,10).fill_(10), 'bias': torch.ones(10).fill_(10)}
        # eval_dict = {'X': torch.ones(100, 784).fill_(10),'X_getitem_0':torch.ones(100,784),'weight': torch.ones(784, 10).fill_(10),
        #              'bias': torch.ones(10).fill_(10)}

        getKeys = AllKeys()
        outputDict = getKeys.doit(dag)

        eval_dict = outputDict
        eval_dict['X']=torch.ones(100,784).fill_(10)
        eval_dict['weight'] = torch.ones(784,10).fill_(10)
        eval_dict['bias'] = torch.ones(10).fill_(10)

        print(eval_dict)

        evaluator.eval(**eval_dict)

        range_bits, filtered_vars = self.calculateRange(evaluator, outputs)
        print("range_bits",range_bits)
        print(filtered_vars)
        bfo = BitFlowOptimizer(evaluator, outputs)
        bfo.calculateInitialValues()
        return bfo, range_bits, filtered_vars

    def createExecutableConstraintFunctions(self, area_fn, error_fn, filtered_vars):
        exec(f'''def AreaOptimizerFn(W):
             {','.join(filtered_vars)} = W
             return  {area_fn}''', globals())

        exec(f'''def ErrorConstraintFn(x):
             {','.join(filtered_vars)} = x
             return  {error_fn}''', globals())

    def initializeData(self, model, training_size, testing_size, weight_size, output_size, data_range, range_bits, batch_size):
        data_params = dict(
            batch_size=batch_size
        )

        # generate testing/training data
        training_set = self.gen_data(
            model, training_size, weight_size, output_size, data_range, range_bits)
        train_gen = data.DataLoader(training_set, **data_params)
        test_set = self.gen_data(
            model, testing_size, weight_size, output_size, data_range, range_bits)
        test_gen = data.DataLoader(test_set, **data_params)

        return train_gen, test_gen

    def initializeWeights(self, outputs, num_weights, initial_W):
        # output without grad
        O = torch.Tensor(list(outputs.values()))
        precision = O.clone()

        # weights matrix
        W = torch.Tensor(num_weights).fill_(initial_W)
        init_W = W.clone()

        print(W)
        W += 0

        W.requires_grad = True
        init_W.requires_grad = True

        return O, precision, W, init_W

    # Loss function
    def compute_loss(self, target, y, W, iter, batch_size, precision, epochs, training_size, error_type=1, should_print=True, shouldErrorCheck=False):
        """
        Args:
            error_type:
                1 ==> Paper Error
                2 ==> Soft-Loss on ULP
        """
        unpacked_W = []
        for weight in W:
            unpacked_W.append(weight)

        area = AreaOptimizerFn(unpacked_W)

        if isinstance(target, list):
            target = torch.stack(target)
            y = torch.stack(y)

        loss = 0
        if error_type == 1:

            # Calculate erros
            constraint_err = ErrorConstraintFn(unpacked_W)

            # Sanity error check
            if shouldErrorCheck:
                if self.hasNegatives(constraint_err):
                    raise ValueError(
                        f"ERR NEGATIVE: {constraint_err}, {W}, {area}")

            # If ulp error is reasonable, relax error constraints
            decay = 0.95
            if constraint_err > 0:
                self.prevR = self.R
                self.R *= decay
            else:
                self.R = self.initR

            L2 = torch.sum((y-target.squeeze())**2)

            S = 1
            Q = 100
            loss = (self.R *
                    torch.exp(-1 * constraint_err) + S * area)/batch_size

        else:
            ulp_error = torch.mean(torch.sum(
                self.within_ulp_err(y, target, precision)))
            constraint_err = torch.max(ulp_error.float(), torch.zeros(1))

            # If ulp error is reasonable, relax error constraints
            decay = 0.995
            if torch.abs(ulp_error) < 1:
                self.prevR = self.R
                self.R *= decay
            else:
                self.R = self.initR

            constraint_W = 100 * \
                torch.sum(torch.max(-(W) + 0.5, torch.zeros(len(W))))

            loss = (self.R * ulp_error)/batch_size

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

    def __init__(self, dag, outputs, data_range, training_size=2000, testing_size=200, epochs=10, batch_size=16, lr=1e-4, error_type=1, test_optimizer=True, test_ufb=False):



        # printer=NodePrinter1()
        # printer.run(dag)

        # getKeys=AllKeys()
        # outputDict = getKeys.doit(dag)
        # print(outputDict)

        # Run a basic evaluator on the DAG to construct error and area functions
        bfo, range_bits, filtered_vars = self.constructOptimizationFunctions(
            dag, outputs, data_range)

        # Generate error and area functions from the visitor
        error_fn = bfo.error_fn
        area_fn = bfo.area_fn
        self.createExecutableConstraintFunctions(
            area_fn, error_fn, filtered_vars)

        # Update the dag with round nodes and set up the model for torch training
        dag, weight_size, input_size, output_size = self.update_dag(dag)
        model = self.gen_model(dag)



        # create the data according to specifications
        train_gen, test_gen = self.initializeData(model, training_size, testing_size,
                                                  weight_size, output_size, data_range, range_bits, batch_size)

        # initialize the weights and outputs with appropriate gradient toggle
        O, precision, W, init_W = self.initializeWeights(
            outputs, weight_size, bfo.initial)

        if error_type == 1:
            self.R = 1e5
            self.initR = self.R
            self.prevR = self.R

        # Set up optimizer
        opt = torch.optim.Adam([W], lr=lr)

        # LOAD DATA
        train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # train_gen=train_loader
        # test_gen=test_loader

        # Run training process
        print("\n##### TRAINING ######")
        iter = 0
        for e in range(epochs):
            for t, (inputs, target_y) in enumerate(train_gen):
                print(inputs)
                inputs["W"] = W
                inputs["O"] = O
                y = model(**inputs)

                if isinstance(y, list):
                    y = torch.stack(model(**inputs))
                    target_y = torch.stack(target_y).squeeze()

                loss = self.compute_loss(target_y, y, W, iter, batch_size, precision, epochs, training_size,
                                         error_type=error_type, should_print=True)

                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1

        # Show final results for weight and round them
        print(W)
        W = self.custom_round(W, factor=0.2)
        print(W)

        self.calc_accuracy("TEST", test_gen, W,
                           O, precision, model, False)

        if test_ufb:
            self.calc_accuracy("UFB TEST", test_gen, init_W,
                               O, precision, model, False)

        self.calc_accuracy("TRAIN", train_gen, W,
                           O, precision, model, False)

        if test_ufb:
            self.calc_accuracy("UFB TRAIN", train_gen, init_W,
                               O, precision, model, False)

        print("\n##### MODEL DETAILS #####")
        print(f"ERROR: {ErrorConstraintFn(W.tolist())}")
        print(f"AREA: {AreaOptimizerFn(W.tolist())}")

        if test_optimizer:
            print("\n##### FROM OPTIMIZER ######")
            bfo.solve()
            test = list(bfo.fb_sols.values())
            print(f"ERROR: {ErrorConstraintFn(test)}")
            print(f"AREA: {AreaOptimizerFn(test)}")

            self.calc_accuracy("OPTIMIZER TEST", test_gen, torch.tensor(test),
                               O, precision, model, False)

        # Save outputs to object
        self.model = model
        self.W = W
        self.O = O