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
        #64 bit precision
        W = torch.Tensor(1, size_w).fill_(64)[0]
        print("START")
        for i in range(num):

             #new_x = torch.normal(mean, std, (1, size_x))
             #new_x = torch.rand((1, size_x))
             new_x = torch.tensor((-4) * torch.rand((size_x)) + 2)
             # TODO: generate data based of properties of data_range
            # new_x = torch.tensor([((data_range[0][1] - data_range[0][0]) * torch.rand((1, 1)) + data_range[0][0]).item(
            # ), ((data_range[1][1] - data_range[1][0]) * torch.rand((1, 1)) + data_range[1][0]).item()])

             inputs = {"X": new_x, "W": W, "O": output_precision}
             new_y = model(**inputs)
             # print("newx,newy")
             # print(new_x,new_y)

             X.append(new_x.tolist())
             Y.append(new_y.tolist())

        print("X VALUES")
        print(X)
        print("Y VALUES")
        print(Y)

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
        #return train_Y[i] - y) - 2 ** (-16), torch.zeros(1))
        return (self.round_to_precision(truth, precision) + 2**-(precision + 1) > num and self.round_to_precision(truth, precision) - 2**-(precision + 1) < num)

    def error_value_fn(self, num, truth, precision):
        return abs(truth - num) - 2 ** -(precision + 1)

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

        # Forward pass: compute predicted y by passing x to the model.

        model = self.gen_model(dag)

        # Training details
        training_size = 1000
        testing_size = 200
        input_size = 2  # TODO: adapt to DAG
        weight_size = 6  # TODO: adapt to DAG
        epochs = 10
        lr_rate = 0.000001

        # output without grad TODO: generalize to DAG
        O = torch.tensor([precision])

        # generate testing/training data
        train_X, train_Y = self.gen_data(model, O, training_size,
                                         input_size, weight_size, mean, std)
        print("DATA")
        print(train_X,train_Y)

        test_X, test_Y = self.gen_data(model, O, testing_size,
                                       input_size, weight_size, mean, std)

        # weights matrix TODO: generalize to DAG
        W = torch.tensor([12., 12., 12., 12., 12.], requires_grad=True)

        # Loss function
        def compute_loss(target, y, W, iter):

            area = torch.tensor(AreaOptimizerFn(W.tolist()))

            error_print = 10*precision*(torch.abs(target - y) - 2** -(precision+1))
            print("ERROR PRINT")
            print(target)
            print(y)
            print(error_print)

            #constraint_2 = 500*torch.max(torch.abs(target - y[0]) - 2 ** -(precision + 1), torch.zeros(1))

            # constraint_1 is the total bits should be greater than 0, will need to pass in range array as well
            constraint_weight = 100 * torch.sum(torch.max(-(W) + 0.5, torch.zeros(len(W))))

            constraint_3= torch.max(error_print, torch.zeros(1))

            # TODO: error is within 1 ulp of truth
            error = torch.tensor(ErrorConstraintFn(W.tolist()))  # >= 0

            error_1 = torch.abs(target - y) - 2 ** -(precision + 1)

            #loss = (area + constraint_weight + torch.exp(-1 * constraint_3))
            loss = (area + constraint_weight + constraint_3)

            if constraint_3>0:
                    print(constraint_3)
                    print("ERROR IS LARGE")
                    # print(error_1)
                    # print(constraint_3)
                    # print()
                    # print(constraint_3)
                    # print("LOSS")
                    # print(loss)
                    # print("WEIGHT")
                    # print(W)
            if constraint_weight > 0:
                    print("WEIGHTS ARE NEGATIVE")

            if iter % 100 == 0:
                print(
                    f"iteration {iter} of {epochs * training_size} ({(iter * 100.)/(epochs * training_size)}%)")
                print(f"AREA: {area}")
                print(f"ERROR: {error_print}")
                # print(f"ERROR_OTHER: {error}")
                print(f"LOSS: {loss[0]}")
                print(f"WEIGHT: {W}")

            return loss

        # Set up optimizer
        opt = torch.optim.AdamW([W], lr=lr_rate)
        # Run training process
        print("\n##### TRAINING ######")
        iter = 0
        for e in range(epochs):
            for i in range(training_size):
                #W.clamp(min=-0.4) #NEW!
                inputs = {"X": train_X[i], "W": W, "O": O}
                y = model(**inputs)
                # print("TRAIN Y")
                # print(train_Y[i])
                #
                # print("calculated value")
                # print(y)

                #NEW!
                #y = torch.clamp(y, min=train_Y[i][0]-2**-(precision+1), max=2**-(precision+1)-train_Y[i][0])
                loss = compute_loss(train_Y[i], y, W, iter) #takes in a Y-value(target value), calculated y from model, Weights vector, iteration number

                # print("VALUES!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(train_Y[i],y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1

        W = torch.ceil(W)

        # Run testing process
        success = 0
        success_1 = 0;
        max_error=0
        average_error=0
        print("\n##### ERRORS ######")
        for i in range(testing_size):
            sample_X = test_X[i]
            sample_Y = test_Y[i]
            inputs = {"X": sample_X, "W": W, "O": O}
            res = model(**inputs)
            #      print("error calc")
            #     print(torch.abs(train_Y[i] - y))
            average_error+=abs(res[0]-sample_Y)
            if (abs(res[0]-sample_Y)>max_error):
                max_error=abs(res[0]-sample_Y)
            if (self.is_within_ulp(res, sample_Y, precision-1)):
                success += 1
                print(f"{res[0]} vs {sample_Y}")
            if (abs(res-sample_Y)-2**-(precision+1)<=0):
                success_1 += 1
            else:
                print(f"{res[0]} vs {sample_Y}")


        acc = (success * 1.)/testing_size
        acc_1 = (success_1 * 1.)/testing_size
        average_error = average_error *1./testing_size

        print(f"accuracy: {acc}")
        print(f"accuracy: {acc_1}")
        print(f"average error: {average_error}")
        print(f"max error: {max_error}")

        print("\n##### SAMPLE ######")

        # Basic sample (truth value = 16)
        test = {"X": torch.tensor([4., 4.]), "W": W, "O": O}
        area = torch.tensor(AreaOptimizerFn(W.tolist()))
        print("Area")
        print(area)
        print(W)
        print(model(**test))
        print("area_1")
        area_1 = torch.tensor(AreaOptimizerFn(W.tolist()))
        print(area_1)
        print(abs(res-sample_Y)-2**-(precision-1)<=0)
        #print(self.is_within_ulp(model(**test),torch.tensor([16.]), precision))