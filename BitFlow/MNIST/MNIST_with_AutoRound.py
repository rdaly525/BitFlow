import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import NodePrinter1
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer
from BitFlow.AddRoundNodes import AddRoundNodes

import torch.nn as nn


# to create MNIST dag


def update_dag(dag):
    """
    Args:
        dag: Input dag
    Returns:
        updated dag: A dag which BitFlow can be run on
    """
    P = Input(name="P")
    R = Input(name="R")
    O = Input(name="O")

    # print("DAG before Round Nodes added:")
    # printer = NodePrinter1()
    # printer.run(dag)

    rounder = AddRoundNodes(P,R, O)
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
    row = 100
    col = 10
    size = 784
    dag = gen_linearlayer(row, col, size)

    #newDag = update_dag(dag)
    print("ADD ROUNDED")

    return dag
    #return dag
    #Return dag with round nodes instead
    #return dag






class MNIST_Dag(nn.Module):

    def __init__(self,dag):
        super().__init__()

        # Dimensions for input, hidden and output

        self.output_dim = 10
        self.hidden_dim = 784
        self.batch_size = 100

        self.weight = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))

        self.W = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.O = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))

        self.myparameters = nn.ParameterList([self.weight, self.bias])

    def gen_model(self,**kwargs):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        evaluator = TorchEval(dag)

        #return self.evaluator.eval(X=images, weight=weight, bias=bias, row=input_dim, col=output_dim, size=input_dim)
        #print(kwargs)
        return evaluator.eval(**kwargs)

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_order_derivative(self, s):
        return s * (1 - s)

        # Forward propagation

    def forward(self, X):
        inputs= {"X": X, "weight":self.weight,"bias": self.bias, "row":self.batch_size, "col":self.output_dim, "size": hidden_dim, "W":self.W,"O": self.O}
        #y1 = self.gen_model(X, self.weight, self.bias, self.batch_size, self.output_dim, self.W, self.O)
        y1 = self.gen_model(**inputs)
        return y1

dag = test_linearlayer()
model = MNIST_Dag(dag)
lr_rate = .001
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

batch_size = 100
input_dim = 100
output_dim = 10
hidden_dim = 784

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)

criterion = torch.nn.CrossEntropyLoss()

print("STARTING TRAINING")
iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # inputs["W"] = W
        # inputs["O"] = O

        inputs = {"X":images}
        outputs = model(**inputs)
        #outputs = model(X=images)
        loss = criterion(outputs, labels)

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
