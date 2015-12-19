#CS229 - Project

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
  
from time import time
from subprocess import call
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
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

lang_name_dict = {
   'bg':'Bulgarian',
   'es-ES': 'Spanish(Spain)',
   'my':'Malaysian',
   'sr':'Serbian',
   'xx':'Random',
   'bs':'Bosnian',
   'hr':'Hungary',
   'pt-BR':'Portuguese (Brazil)',
   'cz': 'Czech',
   'id': 'Indonesia',
   'pt-PT':'Portuguese (Portugal)',
   'es-AR': 'Spanish (Argentina)',
   'mk': 'Macedonia',
   'sk': 'Slovakia'
}

def plot_decision_boundary(lang_codes, X, y, pred_func):
  # Set min and max values and give it some padding
  step =0.5 
  x_min, x_max = X[:,0].min() - step, X[:,0].max() + step
  y_min, y_max = X[:,1].min() - step, X[:,1].max() + step
  h = 0.01

  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  print ('predicting')
  # Predict the function value for the whole gid
  Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
  label_to_number = defaultdict(partial(next, count(1)))
  Zenum = np.array([label_to_number[label] for label in Z])
  Zenum = Zenum.reshape(xx.shape)

  yenum = np.array([label_to_number[label] for label in y])  
  print ('predicted')
  
  # Plot the contour and training examples
  fig, ax = plt.subplots() 
  ax.contourf(xx, yy, Zenum, cmap=plt.cm.Spectral)
  ax.scatter(X[:, 0], X[:, 1], c=yenum, s=50*yenum*yenum, cmap=plt.cm.Spectral)
  
  lang_names = get_lang_names(lang_codes)
  lang_names_str = '-'.join(lang_names)

  plt.suptitle(lang_names_str + " - SVD - Downsampled - Logistic Regression")
  
  # axins = zoomed_inset_axes(ax, 2.5, loc=5) # zoom-factor: 2.5, location: upper-left
  # axins.scatter(X[:, 0], X[:, 1], c=yenum, cmap=plt.cm.Spectral)
  # x1, x2, y1, y2 = 0, 2, -1, 1 # specify the limits
  # axins.set_xlim(x1, x2) # apply the x-limits
  # axins.set_ylim(y1, y2) # apply the y-limits
  # plt.title("Zoomed")

def read_data_file(file_name):
    df = pd.read_csv(file_name, encoding='utf-8', sep=r'\t+', engine='python', header=None, names=['text', 'label'])
    print('size of the data = ', df.shape)
    return df

def read_data_files(file_names):
    
    frame = pd.DataFrame()
    data_files = []
    for file_name in file_names:
       data_file = read_data_file(file_name)
       data_files.append(data_file)
    all_data = pd.concat(data_files)
    all_data = all_data.sample(frac=0.25)
    return all_data

def get_vectorizer(vecDict):
    try:
        if vecDict['vid'] == 1:
            vectorizer = CountVectorizer(encoding='unicode', ngram_range=(1, vecDict['ng_max']))
        elif vecDict['vid'] == 2:
            vectorizer = TfidfVectorizer(encoding='unicode', ngram_range=(1, vecDict['ng_max']))
        print(vectorizer)
        return vectorizer
    except:
        print('please specify a valid vectorizer')
        return 0

def vectorize_data(vectorizer, df, fit):
    t0 = time()
    if fit == 1:
        X = vectorizer.fit_transform(df['text'])
    else:
        X = vectorizer.transform(df['text'])
    y = df['label']
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    return (X, y)

def train_model(X, y, clf_class, params):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X, y)
    print("done in %fs" % (time() - t0))
    return clf

def get_data_file_names(env, lang_codes): 
    root_dir = '../data/' + env + '/'
    data_file_names = []
    for lang_code in lang_codes: 
       data_file_names.append(root_dir + lang_code + '.txt')
    return data_file_names

def visualize_language_pairs(env, lang_codes):
    lang_pairs = '_'.join(lang_codes)
    print('Visualizing lang pairs', lang_pairs, '***********************')
    
    data_file_names = get_data_file_names(env, lang_codes)
    data = read_data_files(data_file_names)
    
    vectorizer = get_vectorizer({'vid': 1, 'ng_max': 1})
    (X, y) = vectorize_data(vectorizer, data, 1)
    
    Xdim = apply_svd(X, 2)
    
    parameters = {}       
    clf = train_model(Xdim, y, LogisticRegressionCV, parameters)
   
    plot_decision_boundary(lang_codes, Xdim, y, lambda x: clf.predict(x))
    print('plotted')
    
    root_dir = '../data/' + env + '/'+env+'_data_language_pairs_visualizations/' 
    try:
       os.stat(root_dir)
    except:
       os.mkdir(root_dir)   
   
    plot_file_name = root_dir + env + '_'+ lang_pairs + '.png'
    plt.savefig(plot_file_name)   # save the figure to file

def visualize_languages(env, lang_codes):
    
    for i in range(0, len(lang_codes)):
       language_code1 = lang_codes[i]
       for j in range(0, len(lang_codes)):
          language_code2 = lang_codes[j]
          if i == j:
             continue
          lang_pairs = []
          lang_pairs.append(language_code1)
          lang_pairs.append(language_code2)
          
          visualize_language_pairs(env, lang_pairs)
    
def apply_svd(X, dim):
    print('X.shape', X.shape)
    svd = TruncatedSVD(n_components=dim)
    svd.fit(X)
    Xdim = svd.transform(X)
    print('Xdim.shape', Xdim.shape)

    return Xdim

def get_lang_names(lang_codes):
   lang_names = []
   for lang_code in lang_codes:
      lang_names.append(lang_name_dict[lang_code])
   return lang_names

def main():
    lang_codes= ['bg','es-ES','my','sr','xx','bs','hr','pt-BR','cz','id','pt-PT','es-AR','mk','sk']  
    visualize_languages('test', lang_codes )
    
if __name__=='__main__':
    main()
