#CS229 - Project
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
    cr = classification_report(y, pred)
    print(cr)
    cm = confusion_matrix(y, pred)
    print("Confusion matrix:")
    print(cm)
    return [cm, cr]

def save_cm(cm, filename, target_names):
    fig = plt.figure()
    plt.matshow(cm, cmap=plt.cm.RdPu)
    plt.title("Confusion Matrix - Langid.py")
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

def dump_results(x,y,pred, filename):
    test_result = {}
    test_result['xgold'] = x
    test_result['ygold'] = y
    test_result['pred'] = pred
    f = open(filename,"wb")
    pickle.dump(test_result, f)

def load_results(filename, labels):
    test_result = pickle.load( open( filename, "rb" ) )
    [x, y, pred]  = [ test_result['xgold'], test_result['ygold'], test_result['pred']]
    pred = [preprocess_pred(p, labels) for p in pred]
    return [x, y, pred]

def preprocess_pred(pred, labels):

    if pred == 'cs':
        pred = 'cz'

    if pred == 'ms':
        pred = 'my'

    if pred in labels:
        return pred
    else:
        return 'xx'

def preprocess_gold(x):
    if x in ['es-ES', 'es-AR']:
        return 'es'

    if x in ['pt-PT', 'pt-BR']:
        return 'pt'

    return x

def main():
    lang_codes= ['bg','es-ES','my','sr','bs','hr','pt-BR','cz','es','id','pt-PT','es-AR','mk','sk','xx', 'pt', 'cs']

    goldData = read_data_filtered_by_lang_codes('../data/test/test-gold.txt', lang_codes)

    labels = ['bg', 'bs', 'cz','es', 'esAR', 'esES', 'hr', 'id', 'mk', 'my', 'pt', 'ptBR', 'ptPT', 'sk', 'sr', 'xx']
    [Xgold, ygold, pred] = load_results('output/benchmarks/langid.dat', labels)
    ygold = [preprocess_gold(g) for g in ygold.values]
    [cm, cr] = get_confusion_matrix(pred, ygold)
    labels = ['bg', 'bs', 'cz','es', 'hr', 'id', 'mk', 'my', 'pt', 'sk', 'sr']
    save_cm(cm, 'output/benchmarks/langid_cm.png', labels)


if __name__=='__main__':
    main()
