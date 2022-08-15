import numpy
import pandas
# file = 'prinordata.xlsx'
# data = pandas.read_excel(file,'合并')
# data = data.transpose()
# data = data.values.tolist()
# data = str(data)
# data = data.replace('[','')
# data = data.replace(']','\n')
# data = data.replace(',',' ')
# with open('data/mytrain.txt','w+') as f:
#     f.write(data)
#     f.close()
# file = 'prinordata.xlsx'
# data = pandas.read_excel(file,'test')
# data = data.transpose()
# data = data.values.tolist()
# data = str(data)
# data = data.replace('[','')
# data = data.replace(']','\n')
# data = data.replace(',',' ')
# with open('data/mytest.txt','w+') as f:
#     f.write(data)
#     f.close()
file = 'prinordata.xlsx'
data = pandas.read_excel(file, '合并')
data = data.transpose()
data = data.values.tolist()
data = str(data)
data = data.replace('[', '')
data = data.replace(']', '\n')
data = data.replace(',', ' ')
with open('data/hebing.txt', 'w+') as f:
    f.write(data)
    f.close()