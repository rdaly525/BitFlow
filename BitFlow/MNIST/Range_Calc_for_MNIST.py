import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


import torch


from BitFlow.AddRoundNodes import NodePrinter1
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer

import torch.nn as nn
import math

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.Optimization import BitFlowVisitor, BitFlowOptimizer
from BitFlow.AddRoundNodes import AddRoundNodes

import torch
from torch.utils import data

import random
import math




def update_dag(dag):
    """
    Args:
        dag: Input dag
    Returns:
        updated dag: A dag which BitFlow can be run on
    """
    W = Input(name="W")
    O = Input(name="O")

    # print("DAG before Round Nodes added:")
    # printer = NodePrinter1()
    # printer.run(dag)

    rounder = AddRoundNodes(W, O)
    roundedDag = rounder.doit(dag)

    # print("DAG after Round Nodes added:")
    # printer.run(roundedDag)

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
    row = 4
    col = 2
    size = 2
    dag = gen_linearlayer(row, col, size)

    # print("BEFORE ADD ROUNDED")
    # newDag = update_dag(dag)
    # print("AFTER ADD ROUNDED")

    return dag


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
def calculateRange(evaluator, outputs):
        print("calculateRange", outputs)
        node_values = evaluator.node_values
        print(node_values)
        print('138')
        visitor = BitFlowVisitor(node_values)
        print("140")
        visitor.run(evaluator.dag)
        print("140")

        filtered_vars = filterDAGFromOutputs(visitor, outputs)

        range_bits = visitor.IBs
        return range_bits, filtered_vars

def constructOptimizationFunctions(dag, outputs, data_range):
        evaluator = NumEval(dag)

        eval_dict = data_range.copy()
        print(eval_dict)
        for key in eval_dict:
            print(key)
            # eval_dict[key] = [int(max(abs(eval_dict[key][0]),
            #                          abs(eval_dict[key][1])))]



        print('here')
        print(eval_dict)

        evaluator.eval(**eval_dict)


        print('162')
        print(outputs)
        range_bits, filtered_vars = calculateRange(evaluator, outputs)
        print('164')
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


    # Loss function

#TODO: Call this compute_loss for precision bits
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

class MNIST_Dag(nn.Module):

    def __init__(self, dag):
        super().__init__()

        # Dimensions for input, hidden and output

        self.output_dim = 2
        self.hidden_dim = 2
        self.batch_size = 4

        # For normal MNIST training
        self.weight = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))

        # TODO: Train W and O for precision bits
        # self.W = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.W = nn.Parameter(torch.Tensor(self.hidden_dim, self.output_dim).fill_(20.), requires_grad=True)
        # self.O = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))
        self.O = nn.Parameter(torch.Tensor(self.output_dim).fill_(20.), requires_grad=True)

        self.myparameters = nn.ParameterList([self.weight, self.bias, self.W, self.O])

    def gen_model(self, **kwargs):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        evaluator = TorchEval(dag)

        # return self.evaluator.eval(X=images, weight=weight, bias=bias, row=input_dim, col=output_dim, size=input_dim)
        # print(kwargs)
        return evaluator.eval(**kwargs)

    def forward(self, X):
        inputs = {"X": X, "weight": self.weight, "bias": self.bias, "row": self.batch_size, "col": self.output_dim,
                  "size": hidden_dim, "W": self.W, "O": self.O}
        # y1 = self.gen_model(X, self.weight, self.bias, self.batch_size, self.output_dim, self.W, self.O)

        #y1 is calculated based on input image, weight, bias
        y1 = self.gen_model(**inputs)

        #print(self.W, self.O)

        #TODO: incorporate for self.W and self.O, use them to custom_round nodes

        return y1


dag = test_linearlayer()

printer = NodePrinter1()
printer.run(dag)
# for dag_input in dag.inputs:
#     print(dag_input)

batch_size = 4

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

n_iters = 3000


input_dim = 4
output_dim = 2
hidden_dim = 2


#test case right now is (4,2) X (2,2) + (2,0)
outputs = {'y': torch.ones(output_dim).fill_(9)}
#dag.outputs

#data_range = {'X': [(0.,1.) for i in range(input_dim*hidden_dim)], 'weight': [(0.,9.) for j in range(hidden_dim*output_dim)], 'bias': [(0.,9.) for k in range(output_dim)]}
#data_range = {'X': [1 for i in range(input_dim*hidden_dim)], 'weight': [9 for j in range(hidden_dim*output_dim)], 'bias': [9 for k in range(output_dim)]}
#data_range = {'X': [1, ([1,1] for i in range(input_dim-1))], 'weight': [[9,9] for j in range(hidden_dim)], 'bias': [9 for k in range(output_dim)]}
data_range = {'X': torch.ones(input_dim,hidden_dim).fill_(9), 'weight': torch.ones(hidden_dim,output_dim).fill_(9), 'bias':torch.ones(output_dim).fill_(9)}

# print("weight_PRINT")
# print(data_range['weight'])
# print(data_range['weight'][:,0])

#data_range = {'X0': (0., 1.), 'X1': (0., 1.), 'X2': (0., 1.), 'X3': (0., 1.), 'weight': (0., 10.), 'bias': (0, 10.)}

training_size=len(train_dataset)
testing_size=len(test_dataset)
epochs = n_iters / (len(train_dataset) / batch_size)
lr_rate = .001
error_type=1
test_optimizer=True
test_ufb=False



# Run a basic evaluator on the DAG to construct error and area functions
bfo, range_bits, filtered_vars = constructOptimizationFunctions(
            dag, outputs, data_range)
# Generate error and area functions from the visitor
error_fn = bfo.error_fn
area_fn = bfo.area_fn
createExecutableConstraintFunctions(
    area_fn, error_fn, filtered_vars)

# Update the dag with round nodes and set up the model for torch training
dag, weight_size, input_size, output_size = update_dag(dag)
model = MNIST_Dag(dag)


# # initialize the weights and outputs with appropriate gradient toggle
# O, precision, W, init_W = initializeWeights(
#     outputs, weight_size, bfo.initial)

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

        if iter % 50 == 0:
            print(iter)

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
