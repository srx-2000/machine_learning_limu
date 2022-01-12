import pylab
import torch
from d2l import torch as d2l

batch_size = 256
# 从两个数据集中分别随机的取256个图片
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 定义输入的维度，这里我们使用的是28*28大小图像，而由于要输入到softmax这个网络中，所以
# 需要将这个矩阵拉伸成为一个向量，所以输入层就定义成为了28*28
num_inputs = 28 * 28
# 定义输出的维度，因为这里我们在识别的时候最终会将图片归为10个不同的种类，所以这里就将输出层
# 的个数定义为了10
num_outputs = 10
# 定义一个权重，而这个权重的形状是784行，10列
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 定义一个偏移量，而这个偏移的形状是10列
b = torch.zeros(num_outputs, requires_grad=True)
"""这里先抛出一个大致的想象图：大概处理的过程就是：将一个图片中的每一个像素抠出来，排成一列，
然后，每个像素对应的有一行小数的值，每一行都有10列【也就是说每个像素点都有相应的10个类别的值】，
然后我们需要做的事情就是，确定这些像素点对应各个列的权重与偏移值，并给出模型，然后利用模型预测出
各个类别所占有的比例【所以从本质上来说，每个像素点对应的那一行的十个列的值相加之后必然等于1】，然后
我们通过对这个像素点的不同的类别的值【说白了就是不同像素点属于那种类型的可能性更大一些，因为他们相加等于1，即百分百】，
来确定该像素点到底属于那种类别。但这里也有个问题，就是我们虽然可以分别预测一个图片的784个像素点分别属于
哪个类别，但我们如何将他们组合起来，是单纯的每个取最大，还是有什么算法之类的"""

# 学习效率
lr = 0.1


# 这里优化器依旧使用的是sgd优化器
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# 定义softmax函数
# 经过模型处理之后的矩阵，shape不会变化，但其中的每个元素必定为正数，同时每行相加必定等于1
def softmax(x):
    # 对矩阵去指数，e^x。之所以使用指数进行运算，有两个优点
    # 1.指数可以保证永远为正数
    # 2.原公式是：exp(xi)/sum(exp(xi))，所以从这里我们可以看出，分子相加必然等于分母
    #    说白了也就是相加必然等于100%，这符合分类这个想法的直觉。即各个类别加起来的总概率必然得1
    x_exp = torch.exp(x)
    # 从上面的公式我们可以看出来，分母其实就是分子相加，所以这里我们也对分母做了一个加和。
    # 但需要注意:这里是对第一个维度进行了加和，而维度是从0开始的。同时需要注意，这里keepdim置True了，也就是维度保留了，举个例子：
    # [[1,2,3],[4,5,6]] shape=[2,3]，对0维进行加和等于:[[5,7,9]],对1维进行加和等于:[[6],[15]]
    partition = x_exp.sum(1, keepdim=True)
    # 这里用到了广播机制，还是使用上面的那个例子:
    # 在按照1维进行加和之后,我们得到了分母是:[[6],[15]]，但我们的分子是[[1,2,3],[4,5,6]]
    # 那么此时通过广播机制我们就可以得到:[[1/6,2/6,3/6],[4/15,5/15,6/15]]
    return x_exp / partition


# 使用softmax函数实现softmax回归模型
def net(X):
    # 这里的意思是：首先将传入的X的列重新塑性成为与权重W的行数量相同【W的行就是像素点的个数】的矩阵
    # 然后和W做一个矩阵乘法，乘出来的矩阵的shape应该是：[256【这里的256是：批量大小】,10【这里的10是：种类的数量】]，
    '''这里说白了就是把这256个图片的每个图片都给取了特征了，即得到了这256个图片每个图片在各个类别中的值，只不过这个值还没有经过
    # softmax函数，所以不一定是正的，也不一定相加得1，同时这里也解释了前面我提出的问题，即每个像素点都有了自己在各个种类中的概率后
    # 怎么讲这些概率综合成一个图片，这里给的答案是直接使用矩阵乘法进行计算，这样784个像素点就都用权重W给消去了，从而我们不需要关心细节，
    # 只需要每次根据预测结果调节相应的像素点中权重W的值，即可完成对一个图片的总体预测'''
    # 然后再给每列加上偏移量b，最后放进softmax函数中进行处理，出来的结果应该是形状没变，
    # 其中的每个元素必定为正数，同时每行相加必定等于1
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 损失函数，这里的损失函数使用的是交叉熵函数，具体细节参考：https://zhuanlan.zhihu.com/p/61944055
def cross_entropy(y_hat, y):
    # 首先公式参考上面链接给出的公式，然后是下面这个表达式的意思：
    # 1.y_hat[range(len(y_hat)), y]的意思是在传入的y_hat中取出与传入的y对应的值。
    # 举个例子就是：如果一个图，他的种类有10种，然后，我利用上面的softmax模型预测出了各个种类的概率
    # 然后，我传入真实标签y，那么我就可以用y_hat[range(len(y_hat)), y]这个函数取出这个y对应的概率是多少
    # 2.根据上面链接给出的公式，我们这里对取出个概率做log并取负即可。
    return - torch.log(y_hat[range(len(y_hat)), y])
    # return (y_hat[range(len(y_hat)), y])


# 该函数主要用来计算预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        # 返回的值正确率，如果预测正确了，那么就是1，如果预测错误了，那么就是0，最后一加和并除以y的长度，就可以得到预测正确的概率
        return float(cmp.type(y.dtype).sum()) / len(y)


# 计算在指定数据集上模型的精度,
def evaluate_accuracy(net, data_iter):
    # 判断传入的net是否继承自torch.nn这个模块，如果继承自nn，那么直接启用评估模式即可
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    # 创建累加器实例对象，传入的2应该是训练了两代吧
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]  # 这里其实是调用了__getitem__方法


# 这里说白了就是一个累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 该函数主要作用是实现对传入网络以及优化器的选择，如果传入的网络是使用的torch的相关网络，则直接调用
# 相关函数即可，如果是自己实现的网络那么就用迭代器，不断对train_iter迭代即可，优化器同理
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


#
# # 定义一个动画类用来画动画，一般没有这个需要，可以忽略
# class Animator:
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale="linear", yscale="linear",
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
#         if legend is None:
#             legend = []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale)
#         self.x, self.y, self.fmts = None, None, fmts
#
#     def add(self, x, y):
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print(f"第{epoch+1}代:")
        print("训练集损失：")
        print(train_loss)
        print("训练集准确度：")
        print(train_acc)
        print("测试集准确度：")
        print(test_acc)


# 预测函数
def predict_ch3(net, test_iter, n=6):
    for x, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    d2l.show_images(x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    pylab.show()


if __name__ == '__main__':
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)
