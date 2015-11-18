import numpy as np
import pandas as pd
  
=======
   
>>>>>>> origin/master
from time import time
from subprocess import call

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def read_data(fileName):
    df = pd.read_csv(fileName, encoding='utf-8', sep=r'\t+', engine='python', header=None, names=['text', 'label'])
    print('size of the data = ', df.shape)
    return df

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
        gold_output = 'MNB.txt'
        pd.Series(pred).to_csv(gold_output, index=False)
        #system("python evaluate.py %s test-gold.txt" % gold_output)
        call(["python", "evaluate.py", "%s" %gold_output, "test-gold.txt"])

def main():
    trainData = read_data('train.txt')
    develData = read_data('devel.txt')
    goldData = read_data('test-gold.txt')

    vectorizer = get_vectorizer({'vid': 1, 'ng_max': 1})
    (Xtrain, ytrain) = vectorize_data(vectorizer, trainData, 1)
    (Xgold, ygold) = vectorize_data(vectorizer, goldData, 0)

    print("Testbenching a MultinomialNB classifier...")
    parameters = {'alpha': 0.01}
    clf = train_model(Xtrain, ytrain, MultinomialNB, parameters)
    print(clf)
    benchmark(clf, Xgold, ygold, eval_gold=True)

if __name__=='__main__':
    main()
