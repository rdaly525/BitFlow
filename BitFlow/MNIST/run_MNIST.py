import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torch

from BitFlow.Eval.TorchEval import TorchEval


class MNIST_dag(torch.nn.Module):

    def gen_model(self, dag):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """

        self.evaluator = TorchEval(dag)

        # print("Dag")
        # print(dag)

        def model(images, W, bias, input_dim, output_dim):
            return self.evaluator.eval(X=images, W=W, bias=bias, row=input_dim, col=output_dim, size=input_dim)

        return model

    def __init__(self, dag):

        print("Entered")
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
        lr_rate = 1e-3

        print("here")
        W = torch.ones(input_dim, output_dim, requires_grad=True)

        bias = torch.ones(output_dim, requires_grad=True)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD([W], lr=lr_rate)

        iter = 0
        for epoch in range(int(epochs)):
            for i, (images, labels) in enumerate(train_loader):

                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

                # outputs = model(images)
                # print(bias)
                # print(W)
                outputs = model(images, W, bias, batch_size, output_dim)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                print(W)
                # W.zero_grad()
                # print(loss)
                loss.backward()

                optimizer.step()

                iter += 1
                if iter % 100 == 0:
                    # calculate Accuracy
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = Variable(images.view(-1, 28 * 28))
                        # outputs = model(images)
                        outputs = model(images, W, bias, batch_size, output_dim)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        # for gpu, bring the predicted and labels back to cpu fro python operations to work
                        correct += (predicted == labels).sum()
                    accuracy = 100 * correct / total
                    print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

            self.model = model
            self.W = W
            print(W)
