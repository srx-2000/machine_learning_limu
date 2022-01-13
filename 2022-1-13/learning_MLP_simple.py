import torch
from torch import nn
from d2l import torch as d2l

# 设置超参数
batch_size, lr, num_epoch = 256, 0.1, 10
# 选择交叉熵损失函数【内赠softmax】
loss = nn.CrossEntropyLoss()

# 这里修正一下，之前说nn.Flatten()是在输入层之前创建一个展开层方便定义形状
# 但其实只说对了一半，后来在网络上查了一些代码，发现其基本都是与nn.Linear绑定出现的
# 一般都放在nn.Linear前面。哪怕前面有别的层也不管，只与Linear绑定
"""这里传入的四个参数分别是：
nn.Flatten 和线性层绑定，用来将输入的图片展开为一个28*28,256的矩阵
nn.Linear 输入层的线性矩阵，shape=[28*28,256]
nn.ReLU 定义隐藏层使用的激活函数，这里使用的是ReLU激活函数
nn.Linear 定义输出层的线性矩阵，shape=[256，10]"""
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10))


# 日常用来初始化权重
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    # 扫描整个网络，查看所有是线性的层，并将其的权重初始化为方差为0.01的正态分布随机值
    net.apply(init_weight)
    # 使用SGD优化器
    trainer = torch.optim.SGD(net.parameters(), lr)
    # 加载数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)
