import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
import torch.optim as op


def load_array(dataset, batch_size, is_train=True):
    """构造一个Pytorch数据迭代器."""
    dataset = data.TensorDataset(*dataset)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    batch_size = 10
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    data_iter = load_array((features, labels), batch_size)
    # nn.sequential的作用就是将多层网络按顺序进行排序存放，后续我们可以通过类似于数组的方式对每个层进行访问
    # nn.linear的作用就是建立一个线性模型，传入的两个参数分别是输入的维度，和输出的维度，这里并没有添加隐藏层
    net = nn.Sequential(nn.Linear(2, 1))
    # 这里首先通过net[0]获取到网络中的第0层，即我们放进去的线性模型：nn.Linear(2, 1)
    # 然后通过.weight找到其权重，也就是表达式y=wx+b中的w，然后.data找到其真实数据，并围绕着他的本身的值的正态分布重新给他赋值
    net[0].weight.data.normal_(0, 0.01)
    # 然后通过.bias找到其偏移值，也就是表达式y=wx+b中的b，然后.data找到其真实数据，并重新将其赋值为0
    net[0].bias.data.fill_(0)
    # 这里使用的损失函数是：均方误差，也就是之前自己写的那个：(y_hat - y.reshape(y_hat.shape)) ** 2 / 2 * batch_size
    loss = nn.MSELoss()
    # 这里可以直接使用net.parameters()函数直接获取所有的net中的参数，这个例子中包括了w和b
    # 调用优化器中的SGD【随机梯度下降】优化器进行优化。
    trainer = op.SGD(net.parameters(), lr=0.03)
    num_epochs = 10
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            # 将优化器清零
            trainer.zero_grad()
            # 这里框架自动帮我们做了sum()，所以直接backword()就好了
            l.backward()
            # 调用step函数对模型进行更新
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1},loss {l:f}")
