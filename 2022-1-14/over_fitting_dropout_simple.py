import torch
from torch import nn
from d2l import torch as d2l

dropout1, dropout2 = 0.3, 0.5
# 参数中的三层分别对应了隐藏层1【input-->hidden1】，
# 隐藏层2【hidden1-->hidden2】，输出层【hidden2-->output】
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(),
                    nn.Dropout(dropout2), nn.Linear(256, 10))


def init_weights(m):
    if type(m) == torch.nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(fn=init_weights)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, test_iter, test_iter, loss=nn.CrossEntropyLoss(), num_epochs=10, updater=trainer)
d2l.plt.show()
