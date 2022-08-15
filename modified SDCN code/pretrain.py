import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
import pandas

#torch.cuda.set_device(3)

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,     #初始化，设置矩阵
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)   #500*500
        self.enc_3 = Linear(n_enc_2, n_enc_3)   #500*2000
        self.z_layer = Linear(n_enc_3, n_z)  #2000*10

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):                  #激活函数,搭建dnn模型的自适应编码器
        enc_h1 = F.relu(self.enc_1(x))  #编码
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)
        #print('z：',z,'\n')

        dec_h1 = F.relu(self.dec_1(z))  #解码
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        #print('x_bar：',x_bar,'\n')

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        data = np.transpose(data)
        self.x = data

    def __len__(self):        #变量个数
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):     #改变学习率
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset):     #输出kmeans聚类结果
    train_loader = DataLoader(dataset, batch_size=25, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)    #优化器，根据亚当算法得来，算wb的
    for epoch in range(30):
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)  ##测量均方误差
            optimizer.zero_grad()  #梯度归零
            loss.backward()   #反向传播计算梯度值
            optimizer.step()  #梯度下降，参数更新

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=6, n_init=20).fit(z.data.cpu().numpy())
            with open('kmeasres.txt', 'w+') as f:
                f.write(str(kmeans.labels_))
                f.close()
            #print(kmeans.labels_)
            #eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'mysubwaynor1.pkl')
    # x1 = []
    # x2 = []
    # x3 = []
    # x4 = []
    # x5 = []
    # x6 = []
    # path = 'pridata.xlsx'
    # file = pandas.read_excel(path,'进站数据')
    # file = file['行标签'].tolist()
    # for i in range(0,195):
    #     if (kmeans.labels_[i] == 0):
    #         x1.append(file[i])
    #     if (kmeans.labels_[i] == 1):
    #         x2.append(file[i])
    #     if (kmeans.labels_[i] == 2):
    #         x3.append(file[i])
    #     if (kmeans.labels_[i] == 3):
    #         x4.append(file[i])
    #     if (kmeans.labels_[i] == 4):
    #         x5.append(file[i])
    #     if (kmeans.labels_[i] == 5):
    #         x6.append(file[i])
    # print('x1:', x1)
    # print('x2:', x2)
    # print('x3:', x3)
    # print('x4:', x4)
    # print('x5:', x5)
    # print('x6:', x6)

print(torch.cuda.is_available())
print(torch.cuda.get_device_capability())
device = torch.device('cuda')
print('Using GPU: ', torch.cuda.get_device_name(0))

model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=672,
        n_z=10,).cuda()

x = np.loadtxt('data\\hebing.txt', dtype=float)
x = x.transpose()
#y = np.loadtxt('data//hhar_label.txt', dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset)