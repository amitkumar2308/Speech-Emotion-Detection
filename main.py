import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
import os

#load dataset
paths = []
labels = []
for dirname, _, filenames in os.walk('C:\\Users\\amitk\\Desktop\\SEDetection\\TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        #print(filename)
        label = filename.split('_')[-1]
        #print(label)
        label = label.split('.')[0]
        #print(label.lower())
print("Dataset is Loaded")



