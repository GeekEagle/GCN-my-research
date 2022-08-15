import matplotlib.pyplot as plt
import numpy as np

path = 'F:\\Python code\\venv\\share\man\myproject\mysubwaynor.txt'
rawdata = np.loadtxt(path,dtype=float)
res = 'F:\\Python code\\venv\\share\man\DAEGC-main\\clures.txt'
y = np.loadtxt(res,dtype=int)
plt.figure(figsize=(50,40),dpi=400)
x = list(i for i in range(0,336))
area = np.pi*2**2
for j in range(0,195):
    if (y[j] == 0):
        plt.scatter(x, rawdata[j], area, 'r')
    if (y[j] == 1):
        plt.scatter(x, rawdata[j], area, 'b')
    if (y[j] == 2):
        plt.scatter(x, rawdata[j], area, 'g')
    if (y[j] == 3):
        plt.scatter(x, rawdata[j], area, 'y')
    if (y[j] == 4):
        plt.scatter(x, rawdata[j], area, 'c')
    if (y[j] == 5):
        plt.scatter(x, rawdata[j], area, 'm')
plt.xlabel('time (h)',fontdict={'weight':'normal','size':30})
plt.ylabel('subway data',fontdict={'weight':'normal','size':30})
plt.title('Time array for each station',fontdict={'weight':'normal','size':30})#图的标签
plt.savefig("daegc4.jpg")#保存图像
plt.show()#显示图像
