import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw0(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path,dtype=float)
    enter = rawdata[:,[i for i in range(0,336)]]
    out = rawdata[:,[i for i in range(336,672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res,dtype=int)
    fig = plt.figure(figsize=(40, 30), dpi=400)
    ax = Axes3D(fig)
    x = list(i for i in range(0,336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    for j in range(0,195):
        if (y[j] == 0):
            ax.scatter(x, enter[j], z1,c='r',marker=".")
            ax.scatter(x, out[j], z2,c='r',marker='^')
        if (y[j] == 1):
            ax.scatter(x, enter[j], z1, c='b', marker=".")
            ax.scatter(x, out[j], z2, c='b', marker='^')
        if (y[j] == 2):
            ax.scatter(x, enter[j], z1, c='g', marker=".")
            ax.scatter(x, out[j], z2, c='g', marker='^')
        if (y[j] == 3):
            ax.scatter(x, enter[j], z1, c='y', marker=".")
            ax.scatter(x, out[j], z2, c='y', marker='^')
        if (y[j] == 4):
            ax.scatter(x, enter[j], z1, c='c', marker=".")
            ax.scatter(x, out[j], z2, c='c', marker='^')
        if (y[j] == 5):
            ax.scatter(x, enter[j], z1, c='m', marker=".")
            ax.scatter(x, out[j], z2, c='m', marker='^')
    ax.set_xlabel('time (h)',fontdict={'weight':'normal','size':30})
    ax.set_ylabel('subway data',fontdict={'weight':'normal','size':30})
    ax.set_zlabel('enter or out',fontdict={'weight':'normal','size':30})
    ax.set_title('Time array for each station',fontdict={'weight':'normal','size':30})#图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\sdcn{}.jpg".format(cluster,cluster))#保存图像
    plt.show()#显示图像
def draw1(cluster,argclu):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    enter = rawdata[:, [i for i in range(0, 336)]]
    out = rawdata[:, [i for i in range(336, 672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    fig = plt.figure(figsize=(40, 30), dpi=400)
    ax = Axes3D(fig)
    x = list(i for i in range(0, 336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    color = ['r','b','g','y','c','m']
    for j in range(0, 195):
        if (y[j] == cluster-1):
            ax.scatter(x, enter[j], z1, c=color[cluster-1], marker=".")
            ax.scatter(x, out[j], z2, c=color[cluster-1], marker='^')
    ax.set_xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    ax.set_zlabel('enter or out', fontdict={'weight': 'normal', 'size': 30})
    ax.set_title('Time array for each station type1', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type{}sdcn{}.jpg".format(argclu,cluster,argclu))  # 保存图像
    plt.show()  # 显示图像
def draw2(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    enter = rawdata[:, [i for i in range(0, 336)]]
    out = rawdata[:, [i for i in range(336, 672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    fig = plt.figure(figsize=(40, 30), dpi=400)
    ax = Axes3D(fig)
    x = list(i for i in range(0, 336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    for j in range(0, 195):
        if (y[j] == 1):
            ax.scatter(x, enter[j], z1, c='b', marker=".")
            ax.scatter(x, out[j], z2, c='b', marker='^')
    ax.set_xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    ax.set_zlabel('enter or out', fontdict={'weight': 'normal', 'size': 30})
    ax.set_title('Time array for each station type2', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type2sdcn{}.jpg".format(cluster, cluster))  # 保存图像
    plt.show()  # 显示图像
def draw3(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    enter = rawdata[:, [i for i in range(0, 336)]]
    out = rawdata[:, [i for i in range(336, 672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    fig = plt.figure(figsize=(40, 30), dpi=400)
    ax = Axes3D(fig)
    x = list(i for i in range(0, 336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    for j in range(0, 195):
        if (y[j] == 2):
            ax.scatter(x, enter[j], z1, c='g', marker=".")
            ax.scatter(x, out[j], z2, c='g', marker='^')
    ax.set_xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    ax.set_zlabel('enter or out', fontdict={'weight': 'normal', 'size': 30})
    ax.set_title('Time array for each station type3', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type3sdcn{}.jpg".format(cluster, cluster))  # 保存图像
    plt.show()  # 显示图像
def draw4(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    enter = rawdata[:, [i for i in range(0, 336)]]
    out = rawdata[:, [i for i in range(336, 672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    fig = plt.figure(figsize=(40, 30), dpi=400)
    ax = Axes3D(fig)
    x = list(i for i in range(0, 336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    for j in range(0, 195):
        if (y[j] == 3):
            ax.scatter(x, enter[j], z1, c='y', marker=".")
            ax.scatter(x, out[j], z2, c='y', marker='^')
    ax.set_xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    ax.set_zlabel('enter or out', fontdict={'weight': 'normal', 'size': 30})
    ax.set_title('Time array for each station type4', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type4sdcn{}.jpg".format(cluster, cluster))  # 保存图像
    plt.show()  # 显示图像
def draw5(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    enter = rawdata[:, [i for i in range(0, 336)]]
    out = rawdata[:, [i for i in range(336, 672)]]
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    plt.figure(figsize=(30, 20), dpi=250)
    ax = Axes3D(fig)
    x = list(i for i in range(0, 336))
    z1 = list(0 for i in range(0, 336))
    z2 = list(1 for i in range(0, 336))
    for j in range(0, 195):
        if (y[j] == 4):
            ax.scatter(x, enter[j], z1, c='c', marker=".")
            ax.scatter(x, out[j], z2, c='c', marker='^')
    ax.set_xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    ax.set_zlabel('enter or out', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Time array for each station type5', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type5sdcn{}.jpg".format(cluster, cluster))  # 保存图像
    plt.show()  # 显示图像
def draw6(cluster):
    path = 'F:\\Python code\\venv\\share\man\SDCN\data\\hebing.txt'
    rawdata = np.loadtxt(path, dtype=float)
    res = 'F:\\Python code\\venv\\share\man\SDCN\\res\\clures.txt'
    y = np.loadtxt(res, dtype=int)
    plt.figure(figsize=(40, 30), dpi=400)
    x = list(i for i in range(0, 672))
    area = np.pi * 2 ** 2
    for j in range(0, 195):
        if (y[j] == 5):
            ax.scatter(x, enter[j], z1, c='m', marker=".")
            ax.scatter(x, out[j], z2, c='m', marker='^')
    plt.xlabel('time (h)', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('subway data', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Time array for each station type6', fontdict={'weight': 'normal', 'size': 30})  # 图的标签
    plt.savefig("C:\\Users\\10334\\Desktop\\结果\\{}\\type6sdcn{}.jpg".format(cluster, cluster))  # 保存图像
    plt.show()  # 显示图像

