import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
import os

path = "prinordata.xlsx"
enter = pd.read_excel(path,"进站")
enter = enter.values.tolist()
enter = list(map(list, zip(*enter)))
out = pd.read_excel(path,"出站")
out = out.values.tolist()
out = list(map(list, zip(*out)))
array = np.zeros((2,336))
for i in range(0,195):
    array[0] = enter[i]
    array[1] = out[i]
    outputImg = Image.fromarray(array*255.0)
    outputImg = outputImg.convert('L')
    outputImg.save('gray/sta{}.jpg'.format(i))
outputImg.show()
