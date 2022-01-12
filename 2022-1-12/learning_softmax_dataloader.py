import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib
import pylab


# 该函数的作用是指定四个进程来读取数据，后续写代码的时候可以放到配置文件中去
def get_dataLoader_workers():
    return 4


# 该函数的作用就是生成一个label列表
def get_fashion_mnist_labels(labels):
    test_labels = ["t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [test_labels[int(i)] for i in labels]


# 该方法的作用本质就是将batch_size个图拼接到了一张图进行显示
def show_images(imgs, num_rows, num_cols, titles=None, scale=2.0):
    # 创建一个传入参数大小的画布
    figsize = (num_cols * scale, num_rows * scale)
    # 这里_,的意思是第一个变量是无所谓的变量，所以这里就没有给出命名【其实函数的第一个返回值是画布对象】，主要使用的是第二个变量axes【函数的第二个返回值是画布子图对象】
    # 而此方法的主要作用是将画布进行切割，以方便对子图进行拼接
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    # 将上面函数切割好的子图和真正的图片两两捆绑并显示
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.set_title(titles[i])


# 用来测试数据加载时间的函数
def test_load_time(mnist_train):
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataLoader_workers())
    timer = d2l.Timer()
    for x, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')


# 用来测试图片加载的函数
def test_image_show(mnist_train):
    # 从训练数据中取出batch_size个图片
    x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    pylab.show()


# 用来加载/下载数据的函数
def load_data():
    # 设置使用svg的形式展示图片
    d2l.use_svg_display()
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值聚在0到1之间
    # 我个人的理解是：其实就是通过这个实例将图片转为了一个可以用来计算的浮点tensor，并且给这个tensor做了类似于
    # onehot-encoding,将所有的像素点化为了类似于概率的浮点数。
    trans = transforms.ToTensor()
    # 将训练的数据集下载到data目录下
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    # 将测试的数据集下载到data目录下
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return mnist_train, mnist_test


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 设置使用svg的形式展示图片
    d2l.use_svg_display()
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值聚在0到1之间
    # 我个人的理解是：其实就是通过这个实例将图片转为了一个可以用来计算的浮点tensor，并且给这个tensor做了类似于
    # onehot-encoding,将所有的像素点化为了类似于概率的浮点数。
    trans = transforms.ToTensor()
    # 将训练的数据集下载到data目录下
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    # 将测试的数据集下载到data目录下
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataLoader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataLoader_workers()))


if __name__ == '__main__':
    mnist_train, mnist_test = load_data()
    test_load_time(mnist_train)
