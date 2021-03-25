from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from BitFlow.IA import Interval
from .Eval.TorchEval import TorchEval
from .Optimization import BitFlowVisitor, BitFlowOptimizer

# from .utils import DAGGrapher


import torch

from .AddRoundNodes import AddRoundNodes, LookupTableTransformer
from torch.utils import data

import random
import math


from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


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
import random
import math
import copy
import time
import matplotlib.pyplot as plt




class BitFlow:

    def make_model(self, **kwargs):
        return self.evaluator.eval(**kwargs)

    def gen_model(self, dag):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        self.transformed_dag = dag
        self.evaluator = TorchEval(dag)

        return self.make_model

    def custom_round(self, P, factor=0.5):

        # print(P)
        P = P.tolist()
        for (index, weight) in enumerate(P):
            f, _ = math.modf(weight)
            if f < factor:
                if weight < 0:
                    P[index] = math.ceil(weight)
                else:
                    P[index] = math.floor(weight)
            else:
                if weight < 0:
                    P[index] = math.floor(weight)
                else:
                    P[index] = math.ceil(weight)
        return torch.tensor(P)

    def gen_data(self, model, dataset_size, size_p, size_r, size_output, data_range, true_width=20., dist=0):
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

        return GeneratedDataset(model, dataset_size, size_p, size_r, size_output, data_range, true_width, dist)

    def update_dag(self, dag, graph_loss=False):
        """
        Args:
            dag: Input dag
        Returns:
            updated dag: A dag which BitFlow can be run on
        """
        P = Input(name="P")
        R = Input(name="R")
        O = Input(name="O")

        rounder = AddRoundNodes(P, R, O)
        roundedDag = rounder.doit(dag)

        if graph_loss:
            dag_grapher = DAGGrapher(list(roundedDag.roots()))
            dag_grapher.run(roundedDag)
            dag_grapher.draw()

        print(f"NUMBER NODES: {rounder.num_nodes}")

        return roundedDag, rounder.round_count, rounder.input_count, rounder.output_count, rounder.range_count, rounder.order, rounder.area_weight

    def round_to_precision(self, num, precision):

        if len(precision) > 1:
            num = num.clone()
            scale = 2.0 ** precision

            for (ind, val) in enumerate(scale):

                num[ind] *= val
            num = torch.round(num)
            for (ind, val) in enumerate(scale):
                num[ind] /= val
            return num
        else:
            scale = 2.0 ** precision
            return torch.round(num * scale) / scale


    def round_to_precision(self, num, precision):

        toReturn = []

        # print(num,precision)
        if len(precision) > 1:
            num = num.clone()
            scale = 2.0**precision



            for i in range(len(scale)):
                toReturn.append(torch.round(num*scale[i])/scale[i])

            return toReturn
        else:
            scale = 2.0**precision
            return torch.round(num * scale) / scale


    def is_within_ulp(self, num, truth, precision, should_print=False):

        # print("NUM",num,"TRUTH",truth,"PRECISION",precision)

        a =self.round_to_precision(truth, precision)
        # print("a",len(a))
        # print(num)
        # print(a)
        #
        # print(len(num),len(num[0]))
        # print(len(precision))



        r = [torch.abs(a - b) for a, b in zip(num, a)]
        # print(len(r),len(r[0]))

        # print(r)

        # assert 0
        # r = torch.abs(num - self.round_to_precision(truth, precision))
        ulp = 2**-(precision)
        if len(precision) > 1:
            # sol = torch.ones(r.shape)
            sol = torch.ones(len(r),len(r[0]))
            for (y, row) in enumerate(r):
                for (x, val) in enumerate(row):
                    if val > ulp[y]:
                        if should_print:
                            print(
                                f"guess: {num[y][x]}, true: {truth[y][x]}, ulp: {ulp[y]}")
                        sol[y][x] = 0
            return sol
        else:
<<<<<<< HEAD
            sol = torch.ones(r.shape)
=======
            sol = torch.ones(len(r),len(r[0]))
>>>>>>> 99b41f6a4330b56154ceb92d6180e81aceb8108e
            for (x, val) in enumerate(r):
                if val > ulp[0]:
                    if should_print:
                        print(
                            f"guess: {num[x]}, true: {truth[x]}, ulp: {ulp[0]}")
                    sol[x] = 0
            return sol



    def ulp(self, num, truth, precision):


        rounded = self.round_to_precision(truth, precision)

        row = []
        for i in range(1):
            col = []
            for j in range(10):
                col.append(torch.abs(num[i][j]-rounded[i*j]))
            row.append(col)

        row = torch.tensor(row)

        r = row

        if len(precision) > 1:

            r = r * 2  ** precision

            return r

        else:
            r = r * 2 ** precision
            return r


    def calc_ulp(self, name, test_gen, P, R, O, model):
        ulp_err = 0
        total = 0

        max_ulp = -math.inf

        print(f"\n##### {name} ULP ERROR ######")
        for t, (inputs, Y) in enumerate(test_gen):
            inputs["P"] = P
            inputs["R"] = R
            inputs["O"] = O


            res = model(**inputs)
            if isinstance(res, list):
                res = torch.stack(res)
                Y = torch.stack(Y).squeeze()
            else:
                Y = Y.squeeze()

            ulp = self.ulp(res, Y, O)

            if torch.max(ulp) > max_ulp:
                max_ulp = torch.max(ulp)


            ulp_err += torch.sum(ulp)
            total += torch.numel(ulp)

        avg_ulp = ulp_err/total
        print(f"AVG ERR: {avg_ulp}")
        print(f"MAX ERR: {max_ulp}")

    def calc_accuracy(self, name, test_gen, P, R, O, model, should_print):
        success = 0
        total = 0
        print(f"\n##### {name} SET ######")

        #print(P)
        #print(R)
        #print(O)
        for t, (inputs, Y) in enumerate(test_gen):

            # print(P)
            # print(R)
            #
            # print(O)

            self.X = inputs.view(-1, 28 * 28)

            inputs = {"X": self.X, "weight": self.weight, "bias": self.bias}
            # inputs = {k: inputs[k].to(device) for k in inputs}

            # print("INPUTS",inputs)
            # print(inputs.shape)



            # self.X = inputs.view(-1, 28 * 28)

            # inputs = {"X": self.X, "weight": self.weight, "bias": self.bias}

            #
            # #REMOVE
            #
            # for t, (inputs, target_y) in enumerate(train_gen):
            #
            #     self.X = inputs.view(-1, 28 * 28)
            #
            #
            #     inputs = {"X": self.X, "weight": self.weight, "bias": self.bias}
            #     inputs = {k: inputs[k].to(device) for k in inputs}
            #
            #
            #     inputs["P"] = P
            #     inputs["R"] = R
            #     inputs["O"] = O
            # #REMOVE

            # print(inputs.keys())
            inputs["P"] = P
            inputs["R"] = R
            inputs["O"] = O


            # for i in (inputs):
            #     print(i)
            # assert 0
            res = model(**inputs)

            # print("res",res)
            # print("Y",Y)

            # if isinstance(y, list):
            #     y = torch.stack(y)
            #
            #     target_y = torch.stack([target_y]).squeeze().float()



            if isinstance(res, list):
                res = torch.stack(res)
                # print(Y)
                # print(Y.squeeze())
                # Y = torch.stack(Y).squeeze()
                Y = Y.squeeze()
            else:
                Y = Y.squeeze()
            # Y = Y.squeeze()
            # print(Y)
            ulp = self.is_within_ulp(res, Y, O, should_print)
            success += torch.sum(ulp)
            total += torch.numel(ulp)

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


    def constructOptimizationFunctions(self, dag, outputs, data_range, should_graph=False):
        evaluator = NumEval(dag)

        eval_dict = data_range.copy()
        eps_idx = 1
        for key in eval_dict:
            if self.train_MNIST:

                eval_dict = {"X": torch.ones(1,784).fill_(1.), "weight": torch.ones(784,10).fill_(1.), "bias": torch.ones(10).fill_(1.)}

            elif isinstance(eval_dict[key], list):
                for (ind, val) in enumerate(eval_dict[key]):
                    eval_dict[key][ind] = int(max(abs(val[0]),
                                                  abs(val[1])))
            else:
                eval_dict[key] = int(max(abs(eval_dict[key][0]),
                                         abs(eval_dict[key][1])))
        self.eval_dict = eval_dict
        self.outputs = outputs

        # print(outputs)
        if not self.train_MNIST:
            evaluator.eval(**eval_dict)


        bfo = BitFlowOptimizer(evaluator, outputs)
        bfo.calculateInitialValues()

        return bfo


    def createExecutableConstraintFunctions(self, area_fn, error_fns, filtered_vars):
            exec(f'''def AreaOptimizerFn(P):
                  {','.join(filtered_vars)} = P
                  return  {area_fn}''', globals())

            exec(f'''def ErrorConstraintFn(x):
                  {','.join(filtered_vars)} = x
                  return  [{','.join(error_fns)}]''', globals())

            self.lambdas = [1e5] * len(error_fns)


    def initializeData(self, model, training_size, testing_size, num_precision, num_range, output_size, data_range, batch_size, distribution):
        data_params = dict(
            batch_size=batch_size
        )

        # generate testing/training data
        # LOAD DATA
        training_set = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_set = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())



        train_gen = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
        test_gen = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)



        return train_gen, test_gen


        #return train_gen, test_gen

    def initializeWeights(self, outputs, num_weights, num_range, initial_P, train_range=False):
        # output without grad
        O = torch.Tensor(list(outputs.values()))

        # weights matrix
        P = torch.Tensor(num_weights).fill_(initial_P)
        init_P = P.clone()

        R = torch.Tensor(num_range).fill_(initial_P - 2)
        init_R = R.clone()

        P += 0

        P.requires_grad = True
        if train_range:
            R.requires_grad = True


        return O, P, init_P, R, init_R


    def compute_loss(self, target, y, P, R, O, iter, filtered_vars, batch_size, epochs, training_size, error_type=1,
                     should_print=True, shouldErrorCheck=False, train_range=False, incorporate_ulp_loss=False,
                     limit=1e-4, decay=0.95):
        """
        Args:
            error_type:
                1 ==> Paper Error
                2 ==> Soft-Loss on ULP
        """
        unpacked_P = []
        for weight in P:
            unpacked_P.append(weight)

        area = 0
        if train_range:
            ldict = {"P": P, "R": R}
            evaluator = NumEval(self.original_dag)
            evaluator.eval(**self.eval_dict)

            node_values = evaluator.node_values
            visitor = BitFlowVisitor(node_values, calculate_IB=False)
            visitor.run(evaluator.dag)

            range_list = [val for val in list(
                visitor.node_values) if val not in evaluator.dag.outputs]

            area_fn = visitor.area_fn
            exec(f"{','.join(filtered_vars)}=P", ldict)
            ib_vars = [f"{f}_ib" for f in range_list]
            exec(f"{','.join(ib_vars)}=R", ldict)

            exec(f"area = {area_fn}", ldict)

            area = ldict["area"]
        else:
            area = AreaOptimizerFn(unpacked_P)

        if isinstance(target, list):
            target = torch.stack(target)
            y = torch.stack(y)

        loss = 0
        if error_type == 1:

            # If the error is met, turn off the error. If the error is not met, turn on the error.

            # Calculate errors

            constraint_errs = ErrorConstraintFn(unpacked_P)

            # print(constraint_errs)


            ulp_err = torch.sum(self.ulp(y, target, O))

            # Sanity error check
            if shouldErrorCheck:
                if self.hasNegatives(constraint_err):
                    raise ValueError(
                        f"ERR NEGATIVE: {constraint_err}, {P}, {area}")

            # If ulp error is reasonable, relax error constraints
            S = 1. / self.area_weight


            error_sum = 0
            for (ind, constraint_err) in enumerate(constraint_errs):
                if constraint_err > limit and self.lambdas[ind] > 1e-8:
                    self.lambdas[ind] *= decay
                else:
                    self.lambdas[ind] = 1e5
                error_sum = error_sum + \
                            self.lambdas[ind] * \
                            torch.exp(-10000 * (constraint_err - limit))

            loss = (error_sum + S * area + torch.sum(torch.exp(-1000 *
                                                               (R - 1))) + torch.sum(torch.exp(-1000 * P))) / (
                       batch_size)

            if incorporate_ulp_loss:
                loss = loss + ulp_err / batch_size

            if train_range:
                loss = loss + self.evaluator.saturation
                self.evaluator.saturation = 0.

        else:
            ulp_error = torch.mean(torch.sum(
                self.within_ulp_err(y, target, O)))
            constraint_err = torch.max(ulp_error.float(), torch.zeros(1))

            constraint_W = 100 * \
                           torch.sum(torch.max(-(P) + 0.5, torch.zeros(len(P))))

            loss = (self.R * ulp_error) / batch_size

        # Catch negative values for area and weights
        if shouldErrorCheck:
            if self.hasNegatives(area):
                raise ValueError(f"AREA ERR: {P}, {area}")

            if self.hasNegatives(W):
                raise ValueError(f"WEIGHT ERR: {P}, {area}")

        # Print out model details every so often
        if iter % 1000 == 0 and should_print == True:

            print(
                f"iteration {iter} of {epochs * training_size / batch_size} ({(iter * 100.) / (epochs * training_size / batch_size)}%)")
            print(f"AREA: {area}")
            print(f"ERROR: {constraint_errs}")
            print(f"WEIGHTS: {P},\n         {R}")
            if error_type == 1:
                print(f"ERROR CONST: {self.lambdas}")
            print(f"LOSS: {loss}")

        return loss

    def __init__(self, dag, outputs, data_range, training_size=2000, testing_size=200, batch_size=1, lr=1e-4, error_type=1, test_optimizer=True, test_ufb=True, train_range=False, range_lr=1e-4, distribution=0, graph_loss=False, custom_data=None, incorporate_ulp_loss=False):
        self.train_MNIST = True
        torch.manual_seed(42)

        self.t0 = time.time()

        self.original_dag = copy.deepcopy(dag)

        # Run a basic evaluator on the DAG to construct error and area functions
        bfo = self.constructOptimizationFunctions(
            dag, outputs, data_range, should_graph=graph_loss)



        # self.intervals = bfo.intervals
        self.intervals = bfo.intervals

        # Update the dag with round nodes and set up the model for torch training
        dag, num_precision, num_inputs, num_outputs, num_range, ordered_list, area_weight = self.update_dag(
            dag, graph_loss=graph_loss)

        self.area_weight = area_weight

        model = self.gen_model(dag)



        # create the data according to specifications
        train_gen = None
        test_gen = None
        if custom_data is None:
            train_gen, test_gen = self.initializeData(model, training_size, testing_size,
                                                      num_precision, num_range, num_outputs, data_range, batch_size, distribution)
        else:
            train_gen = custom_data[0]
            test_gen = custom_data[1]

        # initialize the weights and outputs with appropriate gradient toggle
        O, P, init_P, R, init_R = self.initializeWeights(
            outputs, num_precision, num_range, bfo.initial, train_range=train_range)

        filtered_vars = []
        for el in ordered_list:
            if el not in outputs:
                filtered_vars.append(el)
        # print(filtered_vars)

        # Generate error and area functions from the visitor
        error_fns = bfo.error_fns
        area_fn = bfo.area_fn

        self.createExecutableConstraintFunctions(
            area_fn, error_fns, filtered_vars)

        # store data to object
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.lr = lr
        self.P = P
        self.R = R
        self.O = O
        self.model = model
        self.batch_size = 1
        self.training_size = training_size
        self.error_type = error_type
        self.test_optimizer = test_optimizer
        self.test_ufb = test_ufb
        self.init_P = init_P
        self.init_R = init_R
        self.bfo = bfo
        self.train_range = train_range
        self.filtered_vars = filtered_vars
        self.range_lr = range_lr
        self.graph_loss = graph_loss
        self.incorporate_ulp_loss = incorporate_ulp_loss
        self.outputs = outputs

        #
        self.output_dim = 10
        self.hidden_dim = 784



        self.X = torch.ones(self.batch_size, self.hidden_dim)

        self.weight = torch.ones(self.hidden_dim, self.output_dim, requires_grad=True)
        self.bias = torch.ones(self.output_dim, requires_grad=True)

    def train(self, epochs=1, eval_bitflow=True):

        # retrieve initialized data from object
        train_gen = self.train_gen
        test_gen = self.test_gen
        P = self.P
        R = self.R
        O = self.O
        model = self.model
        batch_size = self.batch_size
        training_size = self.training_size
        error_type = self.error_type
        test_optimizer = self.test_optimizer
        test_ufb = self.test_ufb
        init_P = self.init_P
        init_R = self.init_R
        lr = self.lr
        bfo = self.bfo
        train_range = self.train_range
        filtered_vars = self.filtered_vars
        range_lr = self.range_lr
        graph_loss = self.graph_loss
        incorporate_ulp_loss = self.incorporate_ulp_loss
        outputs = self.outputs

        # Set up optimizer
        opt = torch.optim.AdamW(
            [{"params": P}, {"params": R, "lr": range_lr}], lr=lr)

        lr_decay = 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=opt, step_size=5, gamma=lr_decay)

        # device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        # Run training process
        loss_values = []
        print("\n##### TRAINING ######")
        iter = 0
        # print(len(train_gen))

        counter = 0

        # assert 0
        for e in range(epochs):
            print("*****************EPOCH NUMBER*****************",e)
            for t, (inputs, target_y) in enumerate(train_gen):
                print("*********************************COUNTER*****************************",counter)

                if counter == 50:
                    #assert 0
                    break
                
                
            
                

                self.X = inputs.view(-1, 28 * 28)


                inputs = {"X": self.X, "weight": self.weight, "bias": self.bias}
                inputs = {k: inputs[k].to(device) for k in inputs}


                inputs["P"] = P
                inputs["R"] = R
                inputs["O"] = O

                # print(inputs.keys())

                y = model(**inputs)

                # print(y)
                #




                if isinstance(y, list):
                    # print("HERE")

                    y = torch.stack(y)

                    target_y = torch.stack([target_y]).squeeze().float()


                # print(y)
                # print(target_y)
                # assert 0

                # break
                loss = self.compute_loss(target_y, y, P, R, O, iter, filtered_vars, batch_size, epochs, training_size,
                                         error_type=error_type, should_print=True, train_range=train_range, incorporate_ulp_loss=incorporate_ulp_loss)



                loss_values.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1
                counter +=1

            scheduler.step()

        if graph_loss:
            plt.plot(loss_values, '#40739e')
            plt.title('4TH DEGREE POLY APPROX')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Loss')
            plt.show()

        # Show final results for weight and round them

        #REMOVE

       # P = torch.tensor([5.9743e+00, 4.5683e+00, 6.7684e+00, 5.7919e+00, 4.5934e+00, 5.5308e+00,
        #                  5.7919e+00, 4.5934e+00, 5.5308e+00, 5.7919e+00, 4.5934e+00, 5.5308e+00,
         #                 5.7919e+00, 4.5934e+00, 5.5308e+00, 5.7919e+00, 4.5934e+00, 5.5308e+00,
          #                5.7919e+00, 4.5934e+00, 5.5308e+00, 5.7919e+00, 4.5934e+00, 5.5308e+00,
           #               5.7919e+00, 4.5934e+00, 5.5308e+00, 5.7919e+00, 4.5934e+00, 5.5308e+00,
            #              5.7919e+00, 4.5934e+00, 5.5308e+00, 2.0443e-03, 5.2847e+00, 6.5930e+00,
             #             5.5825e+00, 5.5772e+00])
       # R = torch.tensor([4.1753, 4.1753, 4.1753, 4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919,
        #                  4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919,
         #                 4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919,
          #                4.1753, 4.1753, 8.5919, 4.1753, 4.1753, 8.5919, 5.0932, 5.0162, 4.1753,
           #               4.1753, 5.1106])

        #REMOVE
        if eval_bitflow:

            # REMOVE


            # REMOVE

            P = self.custom_round(P, factor=0.05)

            # add 1 to account for the necessary sign bit
            R = self.custom_round(R, factor=0.05)

            # P = torch.ceil(P)
            # R = torch.ceil(R)
            print(f"PRECISION: {P}")
            print(f"RANGE: {R}")
            print(f"ORDER: {filtered_vars}")

            print(f"TIME: {time.time() - self.t0} SECONDS ELAPSED")

            LUTTransformer = LookupTableTransformer(P, R, filtered_vars)
            model = self.gen_model(LUTTransformer.doit(
                self.transformed_dag))#, self.intervals)


            print("CALC ACCURACY")
            self.calc_accuracy("TEST", test_gen, P, R,
                               O, model, False)

            self.calc_ulp("TEST", test_gen, P, R,
                          O, model)

            self.calc_accuracy("TRAIN", train_gen, P, R,
                               O, model, False)

            print("\n##### MODEL DETAILS #####")
            print(f"ERROR: {ErrorConstraintFn(P.tolist())}")
            if train_range:


                ldict = {"P": P, "R": R}
                evaluator = IAEval(self.original_dag)
                evaluator.eval(**self.eval_dict)

                node_values = evaluator.node_values
                visitor = BitFlowVisitor(node_values, calculate_IB=False)
                visitor.run(evaluator.dag)

                range_list = [val for val in list(
                    visitor.node_values) if val not in evaluator.dag.outputs]

                area_fn = visitor.area_fn
                exec(f"{','.join(filtered_vars)}=P", ldict)
                ib_vars = [f"{f}_ib" for f in range_list]
                exec(f"{','.join(ib_vars)}=R", ldict)

                exec(f"area = {area_fn}", ldict)

                area = ldict["area"]
                print(f"AREA: {area}")
                print(f"UFB AREA: {AreaOptimizerFn(init_P)}")
            else:
                print(f"AREA: {AreaOptimizerFn(P.tolist())}")
                print(f"UFB AREA: {AreaOptimizerFn(init_P)}")

            print(f"UFB: {bfo.initial} bits")

            if test_optimizer:
                print("\n##### FROM OPTIMIZER ######")

                bfo.solve()
                test = list(bfo.fb_sols.values())
                # print(test)
                ibs = bfo.visitor.IBs
                for out in outputs:
                    ibs.pop(out, None)
                rng = list(ibs.values())
                # print(rng)
                # print(filtered_vars)

                # for (index, val) in enumerate(test):
                #     print(f"{val} vs {P[index]} = {val - P[index]}")

                print(f"ERROR: {ErrorConstraintFn(test)}")
                print(f"AREA: {AreaOptimizerFn(test)}")

                if test_ufb:
                    self.calc_accuracy("UFB TEST", test_gen, init_P, torch.tensor(rng),
                                       O, model, False)

                    self.calc_accuracy("UFB TRAIN", train_gen, init_P, torch.tensor(rng),
                                       O, model, False)

                self.calc_accuracy("OPTIMIZER TEST", test_gen, torch.tensor(test), torch.tensor(rng),
                                   O, model, False)

                self.calc_accuracy("UFB TEST (BITFLOW PRECISION):", test_gen, P, torch.tensor(rng),
                                   O, model, False)

                print(
                    f"AREA with BITFLOW PRECISION and FIXED RANGE: {AreaOptimizerFn(P.tolist())}")

                self.calc_accuracy("UFB TEST (BITFLOW RANGE):", test_gen, torch.tensor(test), R,
                                   O, model, False)

            # Save outputs to object
        self.model = model
        self.P = P
        self.R = R
        self.O = O

    @ staticmethod
    def save(fileName, bitflow_object):
        with open(f'{fileName}.pt', 'wb') as output:
            torch.save(bitflow_object, output)

    @ staticmethod
    def load(fileName):
        with open(f'{fileName}.pt', 'rb') as input:
            loaded = torch.load(input)

            loaded.P = loaded.P.float()
            loaded.R = loaded.R.float()

            loaded.P.requires_grad = True
            loaded.R.requires_grad = loaded.train_range

            return loaded
