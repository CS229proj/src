#CS229 - Project
from rosette.api import API, DocumentParameters
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
   'sk': 'Slovakia'
}

def read_data(fileName):
    df = pd.read_csv(fileName, encoding='utf-8', sep=r'\t+', engine='python', header=None, names=['text', 'label'])
    print('size of the data = ', df.shape)
    return df

def read_data_filtered_by_lang_codes(fileName, lang_codes):
    df = read_data(fileName)
    #df = df.sample(frac=0.005)
    df = df[df['label'].isin(lang_codes)].reindex()
       
    print('size of the data (filtered) = ', df.shape)
    return df

def get_confusion_matrix(pred, y):
    print(classification_report(y, pred))
    cm = confusion_matrix(y, pred)
    print("Confusion matrix:")
    print(cm)
    return cm

def get_google_predictions(X):
    pred = []
    url = 'https://www.googleapis.com/language/translate/v2/detect?key=AIzaSyBz5jW8MBXCfJWsJg2w6MDJPXygo9_8mO0&quotaUser=ebudur&project:prime-micron-115405:10'
    print('Request started')
    i = 0
    for text in X:
        try:
            i = i + 1
            params = dict(q=text)
            r = requests.get(url, params)
            response = json.loads(r.text)
            lang = response['data']['detections'][0][0]['language']
            pred.append(lang)
            sleep(2)
            if (i%50 == 0):
                print(i)
        except Exception as inst:
            print('Exception', i,inst, response)

    print('Request ended')
    return pred

def get_yandex_predictions(X):
    pred = []
    url = 'https://translate.yandex.net/api/v1.5/tr.json/detect?key=trnsl.1.1.20151209T073301Z.9043fb7628616e19.1698c0d517c3484a66efc5f85ab24ebb64d3f8f1'
    print('Request started')
    i = 0
    for textstr in X:
        try:
            i = i + 1
            params = dict(text=textstr)
            r = requests.get(url, params)
            response = json.loads(r.text)
            lang = response['lang']
            pred.append(lang)
            #sleep(0.05)
            if (i%50 == 0):
                print(i)
        except Exception as inst:
            print('Exception', i,inst, response)

    print('Request ended')
    return pred

def get_rosette_predictions(X):
    pred = []
    api = API(user_key="b65b38189be361e200bc5c36e6522cf2")
    params = DocumentParameters()
    print('Request started')
    i = 0
    for textstr in X:
        try:
            i = i + 1
            params["content"] = textstr
            result = api.language(params)
            #result = json.loads(result)
            lang = result['languageDetections'][0]['language']
            pred.append(lang)
            #sleep(0.05)
            if (i%50 == 0):
                print(i)
        except Exception as inst:
            print('Exception', i,inst, result)

    print('Request ended')
    return pred

def save_cm(cm, filename):
    fig = plt.figure()
    plt.matshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(filename)

def dump_results(x,y,pred, filename):
    test_result = {}
    test_result['xgold'] = x
    test_result['ygold'] = y
    test_result['pred'] = pred
    f = open(filename,"wb")
    pickle.dump(test_result, f)

def main():
    lang_codes= ['bg','es-ES','my','sr','bs','hr','pt-BR','cz','id','pt-PT','es-AR','mk','sk']
    #lang_codes= ['bg','mk','sk','cz','my','id']
    #lang_codes= ['bs','sr','hr','es-ES','es-AR', 'pt-PT','pt-BR']
    #lang_codes2= ['bg','sr','pt-BR','cz','id','es-AR']
    #lang_codes= ['bg','sr']
    goldData = read_data_filtered_by_lang_codes('../data/test/test-gold.txt', lang_codes)

    Xgold = goldData['text']
    ygold = goldData['label']


    #pred = get_yandex_predictions(Xgold)
    #dump_results(Xgold, ygold, pred, 'yandex.dat')
    #cm = get_confusion_matrix(pred, ygold)
    #save_cm(cm, 'output/benchmarks/yandex_cm.png')

    #pred = get_google_predictions(Xgold)
    #dump_results(Xgold, ygold, pred, 'google.dat')
    #cm = get_confusion_matrix(pred, ygold)
    #save_cm(cm, 'output/benchmarks/google_cm.png')

    pred = get_rosette_predictions(Xgold)
    dump_results(Xgold, ygold, pred, 'rosette.dat')
    cm = get_confusion_matrix(pred, ygold)
    save_cm(cm, 'output/benchmarks/rosette_cm.png')

if __name__=='__main__':
    main()
