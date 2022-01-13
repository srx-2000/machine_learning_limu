import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
# 依旧使用 fashion_mnist数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 这里定义了输入层，输出层，隐藏层三个层的维度
num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 256
# 设置隐藏层权重W1为随机值的矩阵,shape=[784,256]
W1 = torch.randn(num_inputs, num_hiddens, requires_grad=True)
# 设置偏移值b1，初始值为0的向量，因为这个b是要加到隐藏层后面的，所以是一个1*256的向量
b1 = torch.zeros(num_hiddens, requires_grad=True)
# 设置输出层的权重W2为随机值的矩阵,shape=[256,10]
W2 = torch.randn(num_hiddens, num_outputs, requires_grad=True)
# 设置偏移值b2，初始值为0的向量，因为这个b是要加到输出层后面的，所以是一个1*10的向量
b2 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2]


# 单隐藏层的多层感知机
def single_hidden():
    num_epoch, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr)
    # 这里尝试自己实现了一下损失计算和反向传播
    # for epoch in range(num_epoch):
    #     for x, y in train_iter:
    #         y_hat = net(x)
    #         # 实现上述流程中的softmax步骤
    #         loss_fun = nn.CrossEntropyLoss()
    #         # 调用上面取出的交叉熵+softmax损失函数
    #         loss = loss_fun(y_hat, y)
    #         # 优化器梯度归零
    #         updater.zero_grad()
    #         # 反向传播
    #         loss.sum().backward()
    #         updater.step()
    #     print(f"第{epoch + 1}代：loss {float(loss.sum().mean()):f}")
    d2l.train_ch3(net, train_iter, test_iter, nn.CrossEntropyLoss(), num_epoch, updater)
    d2l.plt.show()


# 使用relu激活函数，这里说白了就是，如果传入的张量X大于0，那么我就返回X，如果小于0，那么返回0
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


'''定义网络，基本流程：
H=σ(W1x+b1)
O=W2H+b2
Y=softmax(O) 这一步后面使用softmax再实现【对应代码中的nn.CrossEntropyLoss()函数】，这里先实现了前两步
'''


def net(X):
    # 首先将X【这里的x应该是一个图片，也就是一个三维的张量】的形状重新塑性为784列的一个矩阵【这里从三维拉成了二维】
    # shape=[batch_size,28*28]
    X = X.reshape((-1, num_inputs))
    '''这里的矩阵乘法可以使用简写代替：X @ W1，下面同理'''
    H = relu(torch.matmul(X, W1) + b1)
    return (torch.matmul(H, W2) + b2)


if __name__ == '__main__':
    single_hidden()
