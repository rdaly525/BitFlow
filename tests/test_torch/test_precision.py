import torch
from torch.utils import data
from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round

def gen_fig3():
    a = Input(name="a")
    b = Input(name="b")
    d_prec = Input(name="d_prec")
    e_prec = Input(name="e_prec")
    c = Constant(4, name="c")
    d = Mul(a, b, name="d")
    d_round = Round(d, d_prec)
    e = Add(d, c, name="e")
    e_round = Round(e, e_prec)
    z = Sub(e, b, name="z")

    fig3_dag = Dag(output=z, inputs=[a,b])
    return fig3_dag


def test_basic():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Dataset(data.Dataset):
        def __init__(self, dataset_size):
            self.X = torch.rand(size=[dataset_size, 2]) - 0.5 + torch.tensor([2.0, 3.0]).reshape([1, 2])
            self.target_y = torch.tensor([11, 7])
        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.target_y

    params = dict(
      batch_size=16
    )
    training_set = Dataset(1000)
    train_gen = data.DataLoader(training_set, **params)
    test_set = Dataset(200)
    test_gen = data.DataLoader(test_set, **params)

    #Create the learnable weights
    W = torch.tensor([-0.4, -0.5], requires_grad=True)

    #This is the full computation we are doing
    def model(X, W) -> 'y':
        assert X.shape[1:] == W.shape
        return X + W**2

    #TODO try to design a loss function so that W ends up resolving to be the non-negative solution [3ish, 2ish]
    #  Hint: You want to design a function so that:
    #  1) When W is negative the loss is high
    #  2) When W is positive the loss is close to 0
    #  3) When W takes a tiny step from negative to less negative (towards positive), the loss decreases
    def compute_loss(y, target, W):
        #L2 norm squared
        #Ross's solution
        target_loss = 1*torch.sum((y-target)**2)
        w_loss = 1000*torch.sum(torch.max(-W + 0.5, torch.zeros(2)))
        loss = target_loss + w_loss
        return loss

    #We want our model to always produce this target

    epochs = 3
    opt = torch.optim.SGD([W], lr=.001)
    losslog=[]
    for e in range(epochs):
        for t, (X, target_y) in enumerate(train_gen):
            y = model(X, W)
            loss = compute_loss(y, target_y, W)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if t%10==0:
                print(loss)

    def almost(x):
        return all(torch.abs(x) < 0.1)
    print("Final Values for W", W)
    assert almost(W-torch.Tensor([3, 2]))
