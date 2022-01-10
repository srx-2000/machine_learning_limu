import random
import torch
from d2l import torch as d2l


# 该函数主要用来伪造一个特征矩阵和一个标签列，注意返回的y是一列也就是特征矩阵中的元素的标签
def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


# 该方法主要作用就是将synthetic_data函数生产的特征矩阵和标签进行随机打乱并取出batch_size个数据作为样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 该函数是用来定义线性模型的
def linreg(x, w, b):
    # 这一句的意思翻译过来就是：return y=ax+b【不过a是一个权重向量w，x是一个特征矩阵】
    # 而torch.matmul函数在此时的作用类似于torch.mm()函数，即将两个矩阵按照线代中的矩阵乘法相乘
    # 返回的值的shape应该是：
    return torch.matmul(x, w) + b


# 该方法是用来计算损失的
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 该方法主要是用来进行模型优化的。
# 参数：
"""
params：是给出的所有参数，作为输入。这里以线性回归作为例子的话，那么此时params这个参数的值
        就应该是，之前通过linreg函数和squared_loss这两个函数算出来的所有 W【权重】和b【偏差】
        而我们这次模拟的线性回归的例子，主要目的就在于计算 W和b。即对于一个线性函数：y=Wx+b 来说
        确定了权重W和偏差b之后就可以预测出y的值了。
        所以我们可以看出此函数的输入，就是我们要预测的结果的函数表达式中的那些参数。
         
lr：是学习效率，主要作用是用来固定每次梯度下降是的速率的，不能太快也不能太慢，
    太快容易出现梯度震荡【即结果在等高线之间乱窜】，
    太慢会导致计算开销暴增，同时可能会出现局部最优的问题
    
batch_size：是用来进行小范围抽取的批量，即每次抽取batch_size大小的数据进行求均值优化
            这是因为如果不定义这个batch_size，而是每次都对所有样本数据进行求均值优化
            那么此时的计算开销会非常大。同样的这个值也不能太大或太小
            如果太大就会导致运算开销暴增，
            如果太小那么抽取到的样本数据太少求得的均值对整体没有代表意义。
"""


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            # 这里的意思是：每次学习的过程：
            # 就是学习效率【往下走多少】乘以梯度【就是矩阵求出的导数，与等高线垂直】除以抽取到的样本数量【也可以理解为取了个小范围的均值】
            # 并用当前的参数减去上述算出来的值，因为如果不是减去而是相加，那么就是梯度上升了，而我么要做的是梯度下降
            # 换句话说，机器学习的过程就是一个梯度下降的过程，并且在每次梯度下降之后为了保证此次梯度的计算影响下次梯度的计算，所以
            # 每次梯度下降计算过后，都需要将参数的梯度归零，即：param.grad.zero_()
            """注意这里直接调用了param.grad，也就是说传进来的param是经过了backward()，反向传播之后的值"""
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# torch.optim.SGD
if __name__ == '__main__':
    tensor_w = torch.tensor([2, -3.4])
    b = 4.2
    features, labels = synthetic_data(tensor_w, b, 1000)
    batch_size = 10
    # 对w进行初始化，也就是说这是训练之前的w，是多少都无所谓，因为后面会不断训练，所以这里初始化了一个随机值
    w = torch.normal(0, 1, size=(2, 1), requires_grad=True)
    # 对b初始化，也就是说这是训练之前的b，是多少也无所谓，所以这里直接就写成0了
    b = torch.zeros(1, requires_grad=True)
    # 定义学习效率
    lr = 0.03
    # 定义训练多少代
    num_epochs = 10
    # 定义使用的训练模型
    net = linreg
    # 定义使用的损失函数
    loss = squared_loss
    for epoch in range(num_epochs):
        # 这里说白了每次去features, labels中随机取10个值【这里其实是按顺序取的，但由于data_iter函数在内部对indices进行了打乱，所以可以理解为随机取，同时不会出现重复数据，因为本质是按顺序取的】
        for x, y in data_iter(batch_size, features, labels):
            # 通过linreg模型计算出预测值的y，然后送到loss函数之后，计算损失值
            # 此时输入的x的shape=(batch_size,2)，输入的w的shape=(2,1)，所以二者进入模型后
            # 得到的loss的shape=(batch_size,1)，是一个向量而非标量
            y_hat = net(x, w, b)
            # 通过输入的预测值与实际值计算得到损失。
            """注意此时得到的l的shape应该是("batch_size"，1)，而非标量，所以需要使用sum()函数进行加和"""
            l = loss(y_hat, y)
            # 进行反向传播，使之前传入模型的w和b得到相应的梯度，以供sgd优化器中调用
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_1 = loss(net(features, w, b), labels)
            print(f'epoch {epoch +1}, loss {float(train_1.mean()):f}')
