from torch import nn
from d2l import torch as d2l
import torch

dropout1, dropout2 = 0.3, 0.5
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256


def dropout_layer(X, dropput: float):
    # 利用断言判定dropout是否在01之间
    assert 0 <= dropput <= 1
    # 如果dropout==1那么就是全部丢弃了，直接返回一个全为0的张量即可
    if dropput == 1:
        return torch.zeros_like(X)
    # 如果dropout==0那么就是全部保留了，直接返回本身即可
    if dropput == 0:
        return X
    # 随机生成一个所有值都在01之间的X形状的张量，然后判断这个张量中的每一个元素是否大于dropout，
    # 如果大于就是True，如果小于就是false，然后利用float函数将True，false转为0.0和1.0
    mask = (torch.randn(X.shape) > dropput).float()
    # 将上面求的mask张量与X做矩阵乘法，最后得到的结果就是一个X形状的，然后部分节点被删除的矩阵了
    # 其中被删除【删除代指元素值为零】的元素的位置就是上面张量中false所在的位置，最后再给所有还
    # 元素值非0的元素扩大一下倍数。
    return mask * X / (1.0 - dropput)


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.is_training = is_training
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        # 这里是第一个hidden层，所以需要使用激活函数
        # 这里先将X重新塑性成为一个有num_inputs列的矩阵，然后作为输入放到linear1中，
        # 并启用激活函数，使其变为非线性函数
        H1 = self.relu(self.linear1(X.reshape((-1, self.num_inputs))))
        # 只有是训练的时候才需要使用dropout函数
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.linear2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.linear3(H2)
        return out


if __name__ == '__main__':
    net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)
    num_epoch, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)
    d2l.plt.show()
