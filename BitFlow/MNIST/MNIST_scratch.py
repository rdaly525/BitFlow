import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Concat, Reduce, Relu, Tanh
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.Optimization import BitFlowVisitor, BitFlowOptimizer
from BitFlow.AddRoundNodes import AddRoundNodes
import torch.nn.functional as F

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer

import torch.nn as nn


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

        self.lr_rate = 1e-8

        train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        batch_size = 100

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Our parameters
        # W
        self.W = torch.randn(self.hidden_dim, self.output_dim)
        # bias
        self.bias = torch.randn(self.output_dim)

        self.batch_size = 100
        self.n_iters = 3000
        self.epochs = self.n_iters / (len(train_dataset) / self.batch_size)

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

        y2 = self.sigmoid(y1)
        y2 = torch.sum(y2, dim=1)

        return torch.softmax(y2, dim=0)

    def backward(self, X, l, y2):
        # Derivative of binary cross entropy cost w.r.t. final output y4
        self.dC_dy2 = y2 - l

        self.dy2_dy1 = self.sigmoid_first_order_derivative(y2)

        self.y2_delta = self.dC_dy2 * self.dy2_dy1

        # print(self.y2_delta.shape)
        self.dC_dw1 = torch.matmul(torch.t(X), self.y2_delta)

        w_array = [self.dC_dw1 for _ in range(output_dim)]
        self.dC_dw1 = torch.stack(w_array, dim=1)

        self.W -= self.lr_rate * self.dC_dw1 * torch.ones(1, 10)

    def train(self, X, l):
        # Forward propagation
        y = self.forward(X)


        self.backward(X, l, y)


model = MNIST_Dag()

# Loss list for plotting of loss behaviour
loss_lst = []

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 100
input_dim = 100
output_dim = 10
hidden_dim = 784

lr_rate = .001

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

batch_size = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28 * 28))
        y = Variable(labels)

        y_hat = model(images)
        cross_entropy_loss = -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

        # We have to take cross entropy loss over all our samples, 100 in this 2-class iris dataset
        mean_cross_entropy_loss = torch.mean(cross_entropy_loss).detach().item()

        # ADDED
        loss = mean_cross_entropy_loss

        iter += 1
        if iter % 10 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28 * 28))
                # outputs = model(images)
                outputs = model(images)

                predicted = outputs.clamp(min=0)
                total += labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss, accuracy))

        model.train(images, y_hat)
