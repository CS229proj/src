#CS229 - Project
from rosette.api import API, DocumentParameters
import seaborn as sns;
import pickle
from time import sleep
import json, requests
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import sys
from time import time
from subprocess import call
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from itertools import count
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import pylab as Plot
import matplotlib.cm as cm
from palettable.tableau import Tableau_20
from matplotlib import animation

lang_name_dict = {
   'bg':'Bulgarian',
   'es-ES': 'Spanish(Spain)',
   'my':'Malaysian',
   'sr':'Serbian',
   'xx':'Random',
   'bs':'Bosnian',
   'hr':'Croatian',
   'pt-BR':'Portuguese (Brazil)',
   'cz': 'Czech',
   'id': 'Indonesia',
   'pt-PT':'Portuguese (Portugal)',
   'es-AR': 'Spanish (Argentina)',
   'mk': 'Macedonia',
   'sk': 'Slovakia'}

def get_confusion_matrix(pred, y):
    cr = classification_report(y, pred)
    print(cr)
    cm = confusion_matrix(y, pred)
    print("Confusion matrix:")
    print(cm)
    return [cm, cr]

def load_results(filename, labels):
    test_result = pickle.load( open( filename, "rb" ) )
    [x, y, pred]  = [ test_result['xgold'], test_result['ygold'], test_result['pred']]
    pred = [preprocess_pred(p, labels) for p in pred]
    return [x, y, pred]

def preprocess_pred(x, labels):
    if x in labels:
        return x

    if x == 'ms':
        return 'my'

    if x == 'cs':
        return 'cz'
    else:
        return 'xx'

def preprocess_gold(x):
    if x in ['es-ES', 'es-AR']:
        return 'es'

    if x in ['pt-PT', 'pt-BR']:
        return 'pt'

    return x

def save_cm(cm, filename, target_names):
    fig = plt.figure()
    plt.matshow(cm, cmap=plt.cm.RdPu)
    plt.title("Confusion Matrix - Google")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    [width, height] = cm.shape;
    diag_sum = 0
    for x in range(width):
        diag_sum = diag_sum + cm[x][x]
        row_sum = np.sum(cm[x][:])
        for y in range(height):

            div_res = cm[x][y] *100.0/row_sum

            if np.isnan(div_res):
                res = '-'
            else:
                if(div_res != 0):
                    res = str(div_res)+'%'
                else:
                    res = '0'

            plt.annotate(str(res), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    print('diag_sum, accuracy', diag_sum, diag_sum*1.0/13000)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
    plt.savefig(filename)

def main():
    lang_codes= ['bg','es-ES','my','sr','bs','hr','pt-BR','cz','es','id','pt-PT','es-AR','mk','sk','xx', 'pt', 'cs']

    labels = ['bg', 'bs', 'cz','es', 'esAR', 'esES', 'hr', 'id', 'mk', 'my', 'pt', 'ptBR', 'ptPT', 'sk', 'sr', 'xx']
    [Xgold, ygold, pred] = load_results('output/benchmarks/google.dat', labels)
    ygold = [preprocess_gold(g) for g in ygold.values]
    [cm, cr] = get_confusion_matrix(pred, ygold)
    labels = ['bg', 'bs', 'cz','es', 'hr', 'id', 'mk', 'my', 'pt', 'sk', 'sr', 'xx']
    save_cm(cm, 'output/benchmarks/google_cm.png', labels)

if __name__=='__main__':
    main()
