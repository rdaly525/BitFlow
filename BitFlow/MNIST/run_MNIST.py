import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Concat, Len, Reduce, Relu, Tanh
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.Optimization import BitFlowVisitor, BitFlowOptimizer
from BitFlow.AddRoundNodes import AddRoundNodes
import torch.nn.functional as F

import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce, Len, Concat
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval
from BitFlow.MNIST.MNIST_library import dot_product, matrix_multiply, linear_layer


# Step 1. Load Dataset
# Step 2. Make Dataset Iterable
# Step 3. Create Model Class
# Step 4. Instantiate Model Class
# Step 5. Instantiate Loss Class
# Step 6. Instantiate Optimizer Class
# Step 7. Train Model


#Matrix multiply
# Input: Matrix of 1 by 400,Matrix multiply of pixel matrix by weight matrix (400 by 10) = (1 by 10)Output size batch by 10 (10 weights)
# Select Node is Done
# Matrix Multiply Done
# reduce done
# Select first row, select first col, multiply, reduce, matrix multiply (for every single batch)
# this is the for operation
#
# separate 1 by 10 bias (add at end to get final value)
# Then put through Relu operation or Tanh
# comparing vs torch.linear function



class MNIST_dag:

    def gen_model(self, dag):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """

        self.evaluator = TorchEval(dag)
        # print("Dag")
        # print(dag)

        def model(images, W, bias, input_dim, output_dim):
             # print("Mode")
             # print(images.shape)
             # print(W.shape)
             # print(bias.shape)
             # print(input_dim)
             # print(output_dim)

             #return torch.softmax(self.evaluator.eval(X=images,W=W.T, bias = bias, row=input_dim, col = output_dim),dim=0)
            return self.evaluator.eval(X=images, W=W, bias=bias, row=input_dim, col=output_dim)

        return model



    def __init__(self, dag):

        model = self.gen_model(dag)


        train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        batch_size = 100

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


        batch_size = 100
        n_iters = 3000
        epochs = n_iters / (len(train_dataset) / batch_size)
        input_dim = 784
        output_dim = 10
        lr_rate = 1e-8


        W = torch.ones(input_dim,output_dim,requires_grad=True)


        bias = torch.ones(output_dim, requires_grad=True)

        # def criterion( X, actual):
        #     # print(X.shape)
        #     # print(actual.shape)
        #     # print(X)
        #     # print(actual)
        #     return torch.abs(X-actual)

        #criterion = torch.nn.CrossEntropyLoss()
        print(W)
        criterion = torch.nn.CrossEntropyLoss(weight = W.detach())

        #criterion = torch.nn.CrossEntropyLoss(
            #ignore_index=10, weight=W.detach(), reduction='none')
        optimizer = torch.optim.SGD([W,bias], lr=lr_rate)

        iter = 0
        for epoch in range(int(epochs)):
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 28 * 28))


                labels = Variable(labels)

                # print("Ima")
                #
                # print(images.shape)
                # print(labels.shape)

                optimizer.zero_grad()
                outputs = model(images,W,bias,batch_size,output_dim)
                #print(outputs)

                #print(torch.sum(outputs))

                # print(images.shape)
                # print()
                # print(outputs.shape)
                #loss = torch.mean(-torch.sum(images * torch.log(F.softmax(outputs, dim=1)), dim=1))
                #loss = nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)
                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs,labels)
                print(loss)
                # print(loss)
                # print("wo")
                loss.backward()

                optimizer.step()
                # print("hello")
                iter+=1
                if iter%100==0:
                    # calculate Accuracy
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = Variable(images.view(-1, 28*28))

                        outputs = model(images, W, bias, batch_size, output_dim)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        # for gpu, bring the predicted and labels back to cpu fro python operations to work
                        correct += (predicted == labels).sum()
                    accuracy = 100 * correct / total
                    print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

           #              predicted = torch.max(outputs, 0)
           #              predicted = torch.round(outputs.T)
           #              total+= labels.size(0)
           #
           # correct += (predicted == labels).sum()
           #          accuracy = 100 * correct/total
           #
           #          print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss, accuracy))
           #
