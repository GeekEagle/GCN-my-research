from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
import pandas
import drawres
# torch.cuda.set_device(1)

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1) #336 500
        self.enc_2 = Linear(n_enc_1, n_enc_2) #500 500
        self.enc_3 = Linear(n_enc_2, n_enc_3) #500 2000
        self.z_layer = Linear(n_enc_3, n_z) #2000 10

        self.dec_1 = Linear(n_z, n_dec_1) #10 2000
        self.dec_2 = Linear(n_dec_1, n_dec_2) #2000 500
        self.dec_3 = Linear(n_dec_2, n_dec_3) #500 500
        self.x_bar_layer = Linear(n_dec_3, n_input) #500 336

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,#500
            n_enc_2=n_enc_2,#500
            n_enc_3=n_enc_3,#2000
            n_dec_1=n_dec_1,#2000
            n_dec_2=n_dec_2,#500
            n_dec_3=n_dec_3,#500
            n_input=n_input,#672
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))  #训练的参数

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))  #uj
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        #print('forward adj',adj)
        sigma = 0.5
        # GCN Module
        h = self.gnn_1(x, adj)
        print(h.shape)
        julei = h
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)   #Z(l-1):h
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)    #h:z
        predict = F.softmax(h, dim=1)  #激活函数

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)  #样本i属于j类的概率
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z,julei


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
   # print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    #eva(y, y_pred, 'pae')
    #subres = open('subres.txt', 'w+')
    for epoch in range(200):     #迭代
        # if epoch % 1 == 0:
        #     update_interval
            # _, tmp_q, pred, _,julei = model(data, adj)   #算dnn的代表和预测
            # print(adj)
            # tmp_q = tmp_q.data    #计算分布q
            # p = target_distribution(tmp_q)     #计算目标分布p
            #
            # res1 = tmp_q.cpu().numpy().argmax(1)  # Q   三种聚类方法
            # res2 = pred.data.cpu().numpy().argmax(1)  # Z
            # res3 = p.data.cpu().numpy().argmax(1)  # P

            # eva(res3, res1, str(epoch) + 'Q')
            # eva(res3, res2, str(epoch) + 'Z')
            #eva(y, res3, str(epoch) + 'P')

#自监督：p带动q和z，p受q的影响
        x_bar, q, pred, _,julei = model(data, adj)
        tem_q = q
        q = q.data  # 计算分布q
        res2 = pred.data.cpu().numpy().argmax(1)  # Z
        p = target_distribution(q)     #计算目标分布p

        kl_loss = F.kl_div(tem_q.log(), p, reduction='batchmean')  #Lclu
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')  #Lgcn
        re_loss = F.mse_loss(x_bar, data)  #Lres

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    file = open('predict.txt', mode='w')
    # wr = torch.tensor(pred,float)
    wr = julei.data.tolist()
    strNums = [str(wr_i) for wr_i in wr]
    str1 = "\n".join(strNums)
    str1 = str1.replace('[', '')
    str1 = str1.replace(']', '')
    str1 = str1.replace(',', ' ')
    str1 = str1.replace('\'', '')
    # str1 = str1.replace('\n', ' ')
    file.write(str1)
    res = [str(res_i) for res_i in res2]
    res = str(res)
    res = res.replace(']', '')
    res = res.replace('[', '')
    res = res.replace(',','')
    res = res.replace("'",'')
    res = res.replace('\n', '')
    with open('res/clures.txt', 'w+') as f:
        f.write(res)
        f.close()
    print(res)
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    path = "pridata.xlsx"
    data = pandas.read_excel(path,'进站数据')
    data = data['行标签'].tolist()
    res = []
    for i in range(0,len(res2)):
        if(res2[i]!=' '):
            res.append(int(res2[i]))
    drawres.draw0(args.n_clusters)
    for i in range(1,args.n_clusters+1):
        drawres.draw1(i,args.n_clusters)
    # drawres.draw2(args.n_clusters)
    # drawres.draw3(args.n_clusters)
    # drawres.draw4(args.n_clusters)
    # drawres.draw5(args.n_clusters)
    # drawres.draw6(args.n_clusters)
    for i in range(0,195):
        if (res[i] == 0):
            x1.append(data[i])
        if (res[i] == 1):
            x2.append(data[i])
        if (res[i] == 2):
            x3.append(data[i])
        if (res[i] == 3):
            x4.append(data[i])
        if (res[i] == 4):
            x5.append(data[i])
        if (res[i] == 5):
            x6.append(data[i])
    print('x1:', x1)
    print('x2:', x2)
    print('x3:', x3)
    print('x4:', x4)
    print('x5:', x5)
    print('x6:', x6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='hebing')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'mysubwaynor1.pkl'
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    if args.name == 'hebing':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 2
        args.n_input = 672

    print(args)
    train_sdcn(dataset)