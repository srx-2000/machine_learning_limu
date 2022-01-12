import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# PyTorch 不会隐式的调整输入的形状
# 因此，我们定义了展平层(flatten) 在现行层前调整网络输入的形状
# 即在输入层前加上一层，用来定义我们自己想要的输入形状
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


# 这里主要的作用就是扫描所有的层，如果被扫描的层是线性的，
# 那么我们就把这层的权重重新赋值为一个方差为0.01的随机值
# 而我们上面又将输入层前面定义了一个线性层，所以说白了就是给输入做了一个随机初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
# 使用交叉熵损失函数，这里同时还集成了softmax函数，
# 所以这也就是为什么下面传入的net不需要重新使用softmax模型走一遍了
loss = nn.CrossEntropyLoss()
# 使用SGD优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
# 这里调用之前的训练模型进行，训练，传入的参数分别对应：
# 1.网络【这里之所以只用传入简单的线性网络，是因为softmax模型已经集成在了CrossEntropyLloss激活函数中】
# 2.训练集的迭代器
# 3.测试集的迭代器
# 4.损失函数【在这里同时也担任了softmax的激活函数】
# 5.训练的代数
# 6.优化器，这里使用的SGD优化器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
