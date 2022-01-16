import torch
import os
import pandas as pd
from torch import nn
from d2l import torch as d2l
from torch.utils import data

train_data = pd.read_csv(os.getcwd() + os.sep + "train.csv")
test_data = pd.read_csv(os.getcwd() + os.sep + "test.csv")

# 将训练数据和测试数据进行拼接同时，将数据中的index【即前面的标号1,2,3，4啥的】和train数据中的price给去除
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 找到上一步处理出来的数据中所有不是object属性的值，并把他们的列索引给拿出来放到num_features_index中，
# 方便后续直接通过index取出相应列
num_features_index = all_features.dtypes[all_features.dtypes != "object"].index
# 使用上面找到的那些列索引取出相应的值，并对这些值做一个减去均值然后除以方差的操作。
# 这样所有的数据就都变成了均值为0，方差为1的数据了
all_features[num_features_index] = all_features[num_features_index].apply(lambda x: (x - x.mean()) / (x.std()))
# 将一些为空的数据直接填0，即均值
all_features[num_features_index] = all_features[num_features_index].fillna(value=0)
# 这里的意思是对all_features中不是数字类型的数据使用独热编码，而dummy_na=True的意思是说如果有些项是没有
# 采集到数据的，那么会给他归类为未知这一项。
all_features = pd.get_dummies(all_features, dummy_na=True)

# 因为之前是将训练数据和测试数据拼接到一起进行处理的，所以这里获取到train数据的行数
train_row_num = train_data.shape[0]
# 获取训练集的特征
train_features = torch.tensor(all_features.iloc[:train_row_num, :].values, dtype=torch.float32)
# 获取测试集的特征
test_features = torch.tensor(all_features.iloc[train_row_num:, :].values, dtype=torch.float32)
# 获取标签
label = torch.tensor(train_data["SalePrice"].values.reshape(-1, 1), dtype=torch.float32)
# print(train_features.shape)

# 调用均方损失函数
loss = nn.MSELoss()
net = nn.Sequential(nn.Linear(train_features.shape[1], 1))


# 该函数主要的作用以及对应公式，见链接中第二个图片：https://blog.csdn.net/qq_24671941/article/details/95868747
def log_rmse(net, features, labels):
    # 这里的意思是：将features放入到net中进行训练，然后将得到的张量中的每一个变量的值
    # 都限定在1到infinity之间。
    # 其中torch.clamp()函数的作用是将第一个参数指定的矩阵中的每一个值都限定在第二个和第三个参数之间
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    # 这一步对应的公式是：log(y_hat)-log(y)
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 定义一个训练集迭代器，并打乱顺序，每次迭代器中取出的数据个数是batch_size个
    train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x), y)
            l.backward()
            optimizer.step()
        # 这里之所以使用list去存储训练和测试的损失，而不是直接返回相应的值，是因为后面需要用这两个list画图
        # 在k_fold()函数中，我们也可以看到，其实除了画图以外，每次使用训练损失和测试损失的时候都是直接取他们的-1个【最后一个】
        # 所以如果不是为了画图的话，其实直接返回相应的值即可，不用每次加到列表里再返回列表。
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        # print(train_ls[-1])
        # print(test_ls[-1])
    return train_ls, test_ls


# 该函数主要作用就是利用传入的k来将数据集x，y进行对折，每一折的大小是：x.shape[0]//k
# 然后利用for循环遍历整个数据集k，当j==i时【传入的i是用来指定到底使用第几段来当验证集的，所以i==j
# 就证明此时j这段就要被选为成为验证集了】。这一段对应的x和y就会被放到x_valid和y_valid中当做验证集
# 而当i!=j时，就会将x和y分别放到x_train和y_train中当做训练集
def get_k_fold_data(k, i, x, y):
    assert k > 1
    # 一折的大小
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        # 这里其实就是做了一个切分，提前将要切分的段定义出来而已
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat([x_train, x_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1, )), [train_ls, valid_ls], xlabel="epoch", ylabel="rmse",
                     xlim=[1, num_epochs], legend=["train", "valid"], yscale="log")
        print(f"fold {i + 1},train log rmse {float(train_ls[-1])}，"
              f"valid log rmse {float(valid_ls[-1]):f}")
    return train_l_sum / k, valid_l_sum / k


if __name__ == '__main__':
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_1, valid_1 = k_fold(k, train_features, label, num_epochs, lr, weight_decay, batch_size)
    print(f"{k}-折验证：平均训练log rmse：{float(train_1):f},"
          f"平均验证log rmse：{float(valid_1):f}")
    d2l.plt.show()

