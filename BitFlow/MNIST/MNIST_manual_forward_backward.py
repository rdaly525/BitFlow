import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval

from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer


train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class MNIST(torch.nn.Module):

    def __init__(self, dag):
        super().__init__()
        self.evaluator = TorchEval(dag)


    def forward(self, images, W, bias, input_dim, output_dim):

        self.linear = self.evaluator.eval(X=images, W=W, bias=bias, row=batch_size, col=output_dim, size=input_dim)
        return self.linear

batch_size = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 1e-8

W = torch.ones(input_dim, output_dim, requires_grad=True)
bias = torch.ones(output_dim, requires_grad=True)

def gen_linearlayer(row,col, size):
    X = Input(name="X")
    W = Input(name="W")
    bias = Input(name="bias")
    y = linear_layer(X,W,bias,row,col,size)

    fig = Dag(outputs=[y], inputs=[X,W,bias])
    return fig

def test_linearlayer():
    row = batch_size
    col = output_dim
    size = input_dim

    dag = gen_linearlayer(row,col,size)
    return dag



criterion = torch.nn.CrossEntropyLoss()

model = MNIST(test_linearlayer())
optimizer = torch.optim.SGD([W], lr=lr_rate)

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        outputs = model(images, W, bias, batch_size, output_dim)

        y_check = images@W + bias


        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iter += 1
        if iter % 100 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28 * 28))

                outputs = model(images, W, bias, batch_size, output_dim)


                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

