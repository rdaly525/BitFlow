import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval

from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer

import torch.nn as nn


# to create MNIST dag
def gen_linearlayer(row, col, size):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X, W, bias, row, col, size)

    fig = Dag(outputs=[y], inputs=[X, W, bias])
    return fig


def test_linearlayer():
    row = batch_size
    col = output_dim
    size = input_dim

    dag = gen_linearlayer(row, col, size)
    return dag


class MNIST_Dag(nn.Module):

    def __init__(self, ):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = 100
        self.output_dim = 10
        self.hidden_dim = 784

        self.W = nn.Parameter(torch.ones(self.hidden_dim, self.output_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.output_dim, requires_grad=True))

        self.myparameters = nn.ParameterList([self.W, self.bias])

    def gen_model(self, images, W, bias, input_dim, output_dim):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        self.evaluator = TorchEval(test_linearlayer())

        return self.evaluator.eval(X=images, W=W, bias=bias, row=input_dim, col=output_dim, size=input_dim)

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_order_derivative(self, s):
        return s * (1 - s)

        # Forward propagation

    def forward(self, X):
        y1 = self.gen_model(X, self.W, self.bias, self.batch_size, self.output_dim)
        return y1


model = MNIST_Dag()
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

        outputs = model(images)
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
