import torch
from torch.utils import data

def test_basic():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Dataset(data.Dataset):
        def __init__(self, dataset_size):
            self.X = torch.rand(size=[dataset_size, 2]) + torch.tensor([2, 3]).reshape([1, 2])

        def __len__(self):
            return len(self.X)

        def __getitem__(self,index):
            return self.X[index]

    params = dict(
      batch_size=4
    )
    training_set = Dataset(1000)
    train_gen = data.DataLoader(training_set, **params)
    test_set = Dataset(200)
    test_gen = data.DataLoader(test_set, **params)

    #Create the learnable weights
    W = torch.tensor([0.1, -0.1], requires_grad=True)

    #This is the full computation we are doing
    def model(X, W) -> 'y':
        assert X.shape[1:] == W.shape
        return X + W

    def compute_loss(y, target):
        return torch.sum((y-target)**2)

    #We want our model to always produce this target
    target = torch.tensor([10, 6])

    epochs = 2
    opt = torch.optim.SGD([W], lr=.01)
    losslog=[]
    for t, X in enumerate(train_gen):
        y = model(X, W)
        loss = compute_loss(y, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t%10==0:
            print(loss)

    print("Final Values for W", W)
