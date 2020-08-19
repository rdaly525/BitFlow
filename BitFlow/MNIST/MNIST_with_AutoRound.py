import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
# from DagVisitor import Visitor

import torch

from BitFlow.AddRoundNodes import NodePrinter1, MNIST_area
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


# def update_dag(dag):
#     """
#     Args:
#         dag: Input dag
#     Returns:
#         updated dag: A dag which BitFlow can be run on
#     """
#     W = Input(name="W")
#     O = Input(name="O")
#
#     rounder = AddRoundNodes(W, O)
#     roundedDag = rounder.doit(dag)
#
#     print(rounder.round_count, rounder.input_count, rounder.output_count)
#     return roundedDag, rounder.round_count, rounder.input_count, rounder.output_count
#
#     #return roundedDag
def update_dag(dag):
    """
    Args:
        dag: Input dag
    Returns:
        updated dag: A dag which BitFlow can be run on
    """
    W = Input(name="W")
    O = Input(name="O")

    print("DAG before Round Nodes added:")
    printer = NodePrinter1()
    printer.run(dag)

    rounder = AddRoundNodes(W, O)
    roundedDag = rounder.doit(dag)

    print("DAG after Round Nodes added:")
    printer.run(roundedDag)

    return roundedDag

    # return roundedDag, rounder.round_count, rounder.input_count, rounder.output_count


def gen_linearlayer(row, col, size):
    X = Input(name="X")
    weight = Input(name="weight")
    bias = Input(name="bias")
    y = linear_layer(X, weight, bias, row, col, size)

    fig = Dag(outputs=[y], inputs=[X, weight, bias])
    return fig


def test_linearlayer():
    row = 100
    col = 10
    size = 784
    dag = gen_linearlayer(row, col, size)

    # print("BEFORE ADD ROUNDED")
    # newDag = update_dag(dag)
    # print("AFTER ADD ROUNDED")

    return dag


def filterDAGFromOutputs(visitor, outputs):

        # Remove output variables from DAG list (these will be our weights)
        vars = list(visitor.node_values)
        for (i, var) in enumerate(vars):
            vars[i] = var.name
        filtered_vars = []
        for var in vars:
            if var not in outputs:
                filtered_vars.append(var)

        return filtered_vars


def createExecutableConstraintFunctions(area_fn, filtered_vars):
    exec(f'''def AreaOptimizerFn(W):
             {','.join(filtered_vars)} = W
             return  {area_fn}''', globals())

class MNIST_Dag(nn.Module):

    def __init__(self, dag):
        super().__init__()

        # Dimensions for input, hidden and output

        self.output_dim = 10
        self.hidden_dim = 784
        self.batch_size = 100

        # For normal MNIST training
        self.weight = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))

        # TODO: Train W and O for precision bits
        # self.W = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.W_prec = nn.Parameter(torch.Tensor(self.hidden_dim, self.output_dim).fill_(20.), requires_grad=True)
        # self.O = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))
        self.O_prec = nn.Parameter(torch.Tensor(self.output_dim).fill_(20.), requires_grad=True)

        self.myparameters = nn.ParameterList([self.weight, self.bias, self.W_prec, self.O_prec])

    def gen_model(self, **kwargs):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        evaluator = TorchEval(dag)

        # return self.evaluator.eval(X=images, weight=weight, bias=bias, row=input_dim, col=output_dim, size=input_dim)
        # print(kwargs)
        return evaluator.eval(**kwargs)

    def numAdd(self):
        # return self.batch_size*(self.output_dim*self.hidden_dim+self.output_dim+1)+self.batch_size+1
        # return self.batch_size*(self.output_dim*self.hidden_dim+self.output_dim+1+1)+1
        # normalized
        # return 1.0*(self.output_dim * self.hidden_dim + self.output_dim + 1 + 1) + 1
        # one add per reduce

        # 2201
        return 1.0 * (self.output_dim + self.output_dim + 1 + 1) + 1

    def numMul(self):
        # return self.batch_size*self.output_dim

        return 1.0 * self.output_dim

    # TODO:Potentially create a list of all current Add node precisions and all current Multiply node precisions, calculate respective total area for all Add nodes, and total area for Multiply nodes, add two values together

    def genArea_MNIST(self):
        # TODO: traverse through MNIST dag
        # each REDUCE NODE is 1 add
        # each CONCAT NODE is # of inputs, adds
        # each MUL NODE is 1 multiply

        # get number of precision bits at each REDUCE, CONCAT and MUL, create list for both
        # get total area_add(REDUCE, CONCAT) and totalarea_mul(MUL)

        evaluator = NumEval(dag)
        node_values = evaluator.node_values
        visitor = node_values
        visitor.run(evaluator.dag)

    def return_weight(self):
        return self.weight

    def return_bias(self):
        return self.bias

    def forward(self, X):

        inputs = {"X": X, "weight": self.weight, "bias": self.bias, "row": self.batch_size, "col": self.output_dim,
                  "size": hidden_dim, "W": self.W_prec, "O": self.O_prec}
        # y1 = self.gen_model(X, self.weight, self.bias, self.batch_size, self.output_dim, self.W, self.O)

        # y1 is calculated based on input image, weight, bias
        y1 = self.gen_model(**inputs)

        # print(self.W, self.O)

        # TODO: incorporate for self.W and self.O, use them to custom_round nodes

        return y1

    # TODO: For working with precision values
    def custom_round(self, W, factor=0.5):
        W = W.tolist()
        for (index, weight) in enumerate(W):
            f, _ = math.modf(weight)
            if f < factor:
                W[index] = math.floor(weight)
            else:
                W[index] = math.ceil(weight)
        return torch.tensor(W)

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
            print("key")
            print(key)
            eval_dict[key] = int(max(abs(eval_dict[key][0]),
                                     abs(eval_dict[key][1])))

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


    def is_within_ulp(self, num, truth, precision):
        r = torch.abs(num - self.round_to_precision(truth, precision))
        ulp = 2 ** -(precision + 1)
        if len(precision) > 1:
            sol = torch.ones(r.shape)
            for (y, row) in enumerate(r):
                for (x, val) in enumerate(row):
                    if val > ulp[y]:
                        sol[y][x] = 0
            return sol
        else:
            return (torch.where(r <= ulp, torch.ones(r.shape), torch.zeros(r.shape)))

    # TODO: Call this compute_loss for precision bits
    def compute_loss(self, target, y, W, iter, batch_size, precision, epochs, training_size, error_type=1,
                     should_print=True, shouldErrorCheck=False):
        """
        Args:
            error_type:
                1 ==> Paper Error
                2 ==> Soft-Loss on ULP
        """
        unpacked_W = []
        for weight in W:
            unpacked_W.append(weight)

        # TODO: Call Area_MNIST function which will calculate area for current DAG values
        #area = AreaOptimizerFn(unpacked_W)
        area = area_fn

        if isinstance(target, list):
            target = torch.stack(target)
            y = torch.stack(y)

        loss = 0

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

        loss = (self.R * ulp_error) / batch_size

    # Catch negative values for area and weights
    # if shouldErrorCheck:
    #     if self.hasNegatives(area):
    #         raise ValueError(f"AREA ERR: {W}, {area}")
    #
    #     if self.hasNegatives(W):
    #         raise ValueError(f"WEIGHT ERR: {W}, {area}")

        # Print out model details every so often
        if iter % 1000 == 0 and should_print == True:
            print(
                f"iteration {iter} of {epochs * training_size / batch_size} ({(iter * 100.) / (epochs * training_size / batch_size)}%)")
            print(f"AREA: {area}")
            print(f"ERROR: {constraint_err}")
            print(f"WEIGHTS: {W}")
            if error_type == 1:
                print(f"ERROR CONST: {self.R}")
            print(f"LOSS: {loss}")

        return loss


dag = test_linearlayer()

batch_size = 100
n_iters = 3000
input_dim = 100
output_dim = 10
hidden_dim = 784

#LOAD DATA
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# data_range = {'X': (0., 256.), 'weight': (0., 11.), 'bias': (0, 11.)}

training_size = len(train_dataset)
testing_size = len(test_dataset)
epochs = n_iters / (len(train_dataset) / batch_size)
lr_rate = .001
error_type = 1
test_optimizer = True
test_ufb = False


def __init__(self, dag, outputs, data_range, training_size=2000, testing_size=200, epochs=10, batch_size=16, lr=1e-4,
             error_type=1, test_optimizer=True, test_ufb=False):


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


    evaluator = NumEval(dag)
    node_values = evaluator.node_values
    print(node_values)
    #print("DAG run to get area:")
    #visitor = MNIST_area(evaluator)
    visitor = MNIST_area(evaluator)
    visitor.run(dag)


    # Generate area functions from the visitor
    area_fn = visitor.area_fn
    node_values = evaluator.node_values
    # Remove output variables from DAG list (these will be our weights)

    outputsA = dag.outputs

    #filtered_vars = filterDAGFromOutputs(visitor, outputsA)

    #createExecutableConstraintFunctions(area_fn, filtered_vars)


    # Update the dag with round nodes and set up the model for torch training

    dag = update_dag(dag)

    model = MNIST_Dag(dag)

    if error_type == 1:
        R = 1e5
        initR = R
        prevR = R

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

    criterion = torch.nn.CrossEntropyLoss()

    print("STARTING TRAINING")
    iter = 0
    for epoch in range(int(epochs)):
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # inputs["W"] = W
            # inputs["O"] = O

            inputs = {"X": images}
            outputs = model(**inputs)
            # outputs = model(X=images)
            loss = criterion(outputs, labels)

            # print(loss)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # calculate Accuracy
                correct = 0
                total = 0

                for images, labels in test_loader:
                    images = Variable(images.view(-1, 28 * 28))
                    # outputs = model(images)
                    outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / total
                print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
                print("weight", MNIST_Dag.return_weight(model))
                print("bias", MNIST_Dag.return_bias(model))
                # for param in model.parameters():
                #
                #     print(model.parameters()[0])
