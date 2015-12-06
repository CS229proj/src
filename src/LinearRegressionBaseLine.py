#CS229 - Project

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

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

def read_data(fileName):
    df = pd.read_csv(fileName, encoding='utf-8', sep=r'\t+', engine='python', header=None, names=['text', 'label'])
    print('size of the data = ', df.shape)
    return df

def read_data_filtered_by_lang_codes(fileName, lang_codes):
    df = read_data(fileName)
    df = df.sample(frac=0.05)
    df = df[df['label'].isin(lang_codes)].reindex()
       
    print('size of the data (filtered) = ', df.shape)
    return df

def get_vectorizer(vecDict):
    try:
        if vecDict['vid'] == 1:
            vectorizer = CountVectorizer(encoding='unicode', ngram_range=(1, vecDict['ng_max']), dtype=np.float32)
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

def benchmark(clf, X, y, eval_gold=False):
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X)
    print("done in %fs" % (time() - t0))

    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y, pred))
     
    cm = confusion_matrix(y, pred)
    print("Confusion matrix:")
    print(cm)

    if eval_gold:
        gold_output = '../data/temp_output/MNB.txt'
        pd.Series(pred).to_csv(gold_output, index=False)
        #system("python evaluate.py %s test-gold.txt" % gold_output)
        call(["python", "evaluate.py", "%s" %gold_output, "../data/test/test-gold.txt"])

def apply_svd(X, dim):
    print('X.shape', X.shape)
    svd = TruncatedSVD(n_components=dim)
    svd.fit(X)
    Xdim = svd.transform(X)
    print('Xdim.shape', Xdim.shape)

    return Xdim

def plot_tsne(Xin,Yin):
    set2 = Tableau_20.mpl_colors
    X = tsne(Xin, 3, 50, 20.0)
    print("plot_tsne: X.shape")
    print(X.shape)
    labels  = Yin.values
    markers = ["o" ,"<", ">", "^", "v", "x" ]
    fig = Plot.figure()
    ax = Axes3D(fig)
    for (i,cla) in enumerate(set(labels)):
       xc = [p for (j,p) in enumerate(X[:,0]) if labels[j]==cla]
       yc = [p for (j,p) in enumerate(X[:,1]) if labels[j]==cla]
       zc = [p for (j,p) in enumerate(X[:,2]) if labels[j]==cla]
       ax.scatter(xc,yc,zc,c=set2[i],label=lang_name_dict[cla], marker=markers[i%len(markers)], s=40)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([np.min(X[:,0])-40, np.max(X[:,0])])
    ax.set_ylim([np.min(X[:,1]), np.max(X[:,1])])
    ax.set_zlim([np.min(X[:,2]), np.max(X[:,2])])
    ax.legend(loc=2)
    Plot.savefig('tsne3D.png')
    Plot.savefig('tsne3D.fig')
    Plot.show()
    
def main():
    #lang_codes= ['bg','es-ES','my','sr','xx','bs','hr','pt-BR','cz','id','pt-PT','es-AR','mk','sk']
    lang_codes= ['bg','es-ES','my','sr','bs','hr','pt-BR','cz','id','pt-PT','es-AR','mk','sk']

    trainData = read_data_filtered_by_lang_codes('../data/train/train.txt', lang_codes)
    # develData = read_data_filtered_by_lang_codes('../data/dev/devel.txt', lang_codes)
    goldData = read_data_filtered_by_lang_codes('../data/test/test-gold.txt', lang_codes)
    
    vectorizer = get_vectorizer({'vid': 1, 'ng_max': 5})
    (Xtrain, ytrain) = vectorize_data(vectorizer, trainData, 1)
    (Xgold, ygold) = vectorize_data(vectorizer, goldData, 0)
    
    x_dim = 100
    Xtrain = apply_svd(Xtrain, x_dim)
    Xgold = apply_svd(Xgold, x_dim)
    print("Xgold.shape:")
    print(Xgold.shape)    
    plot_tsne(Xgold,ygold)
     
    print("Testbenching a LogisticRegressionCV classifier...")
    parameters = {}
    clf = train_model(Xtrain, ytrain, LogisticRegressionCV, parameters)
    print(clf)
    benchmark(clf, Xgold, ygold, eval_gold=True)

def Hbeta(D = np.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta);
        sumP = sum(P);
        H = np.log(sumP) + beta * np.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;

def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

        # Initialize some variables
        print ("Computing pairwise distances...")
        (n, d) = X.shape;
        sum_X = np.sum(np.square(X), 1);
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
        P = np.zeros((n, n));
        beta = np.ones((n, 1));
        logU = np.log(perplexity);

        # Loop over all datapoints
        for i in range(n):

                # Print progress
                if i % 500 == 0:
                        print ("Computing P-values for point ", i, " of ", n, "...")

                # Compute the Gaussian kernel and entropy for the current precision
                betamin = -np.inf;
                betamax =  np.inf;
                Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
                (H, thisP) = Hbeta(Di, beta[i]);

                # Evaluate whether the perplexity is within tolerance
                Hdiff = H - logU;
                tries = 0;
                while np.abs(Hdiff) > tol and tries < 50:

                        # If not, increase or decrease precision
                        if Hdiff > 0:
                                betamin = beta[i].copy();
                                if betamax == np.inf or betamax == -np.inf:
                                        beta[i] = beta[i] * 2;
                                else:
                                        beta[i] = (beta[i] + betamax) / 2;
                        else:
                                betamax = beta[i].copy();
                                if betamin == np.inf or betamin == -np.inf:
                                        beta[i] = beta[i] / 2;
                                else:
                                        beta[i] = (beta[i] + betamin) / 2;

                        # Recompute the values
                        (H, thisP) = Hbeta(Di, beta[i]);
                        Hdiff = H - logU;
                        tries = tries + 1;

                # Set the final row of P
                P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

        # Return final P-matrix
        print ("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)));
        return P;

def pca(X = np.array([]), no_dims = 50):
        """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
        print (X.shape)
        print ("Preprocessing the data using PCA...")
        (n, d) = X.shape;
        
        X = X - np.tile(X[:,0].mean(), (n, 1));
        (l, M) = np.linalg.eig(np.dot(X.T, X));
        Y = np.dot(X, M[:,0:no_dims]);
        return Y;

def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
        """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
        The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

        # Check inputs
        if isinstance(no_dims, float):
                print ("Error: array X should have type float.");
                return -1;
        if round(no_dims) != no_dims:
                print ("Error: number of dimensions should be an integer.");
                return -1;

        print("X.shape tsne_pca")
        print(X.shape)

        # Initialize variables
        X = pca(X, initial_dims).real;
        (n, d) = X.shape;
        max_iter = 1000;
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 500;
        min_gain = 0.01;
        Y = np.random.randn(n, no_dims);
        dY = np.zeros((n, no_dims));
        iY = np.zeros((n, no_dims));
        gains = np.ones((n, no_dims));

        # Compute P-values
        P = x2p(X, 1e-5, perplexity);
        P = P + np.transpose(P);
        P = P / np.sum(P);
        P = P * 4;                                                                      # early exaggeration
        P = np.maximum(P, 1e-12);

        # Run iterations
        for iter in range(max_iter):

                # Compute pairwise affinities
                sum_Y = np.sum(np.square(Y), 1);
                num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
                num[range(n), range(n)] = 0;
                Q = num / np.sum(num);
                Q = np.maximum(Q, 1e-12);

                # Compute gradient
                PQ = P - Q;
                for i in range(n):
                        dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

                # Perform the update
                if iter < 20:
                        momentum = initial_momentum
                else:
                        momentum = final_momentum
                gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
                gains[gains < min_gain] = min_gain;
                iY = momentum * iY - eta * (gains * dY);
                Y = Y + iY;
                Y = Y - np.tile(np.mean(Y, 0), (n, 1));

                # Compute current value of cost function
                if (iter + 1) % 10 == 0:
                        C = np.sum(P * np.log(P / Q));
                        print("Iteration ", (iter + 1), ": error is ", C)

                # Stop lying about P-values
                if iter == 100:
                        P = P / 4;

        # Return solution
        return Y;

if __name__=='__main__':
    main()
