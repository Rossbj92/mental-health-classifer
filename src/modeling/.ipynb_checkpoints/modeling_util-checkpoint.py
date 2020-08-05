import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None

#Visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
plt.style.use('seaborn')

#Fonts for plots
header_font = fm.FontProperties(fname='./Fonts/Montserrat/Montserrat-Regular.ttf', size = 14)
text_font = fm.FontProperties(fname='./Fonts/Lato/Lato-Regular.ttf', size = 14)
cbar_font = fm.FontProperties(fname='./Fonts/Lato/Lato-Regular.ttf', size = 30)

from joblib import load, dump
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle