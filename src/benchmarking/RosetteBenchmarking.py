#CS229 - Project
from rosette.api import API, DocumentParameters
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pycountry

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
            print(lang)
            pred.append(lang)
            #sleep(0.05)
            if (i%50 == 0):
                print(i)
        except Exception as inst:
            print('Exception', i,inst, result)

    print('Request ended')
    return pred

def save_cm(cm, filename, target_names):
    fig = plt.figure()
    plt.matshow(cm, cmap=plt.cm.RdPu)
    plt.title("Confusion Matrix - Rosette Language API")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    [width, height] = cm.shape;
    print(cm.shape)
    diag_sum = 0
    for x in range(width):
        diag_sum = diag_sum + cm[x][x]
        row_sum = np.sum(cm[x][:])
        for y in range(height):

            div_res = cm[x][y]# *100.0/row_sum

            if np.isnan(div_res):
                res = '-'
            else:
                if(div_res != 0):
                    res = str(div_res)#+'%'
                else:
                    res = '0'

            plt.annotate(str(res), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    print('diag_sum, accuracy', diag_sum, diag_sum*1.0/12000)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
    #plt.savefig(filename)

def dump_results(x,y,pred, filename):
    test_result = {}
    test_result['xgold'] = x
    test_result['ygold'] = y
    test_result['pred'] = pred
    f = open(filename,"wb")
    pickle.dump(test_result, f)

def load_results(filename, labels, lang_codes):
    test_result = pickle.load( open( filename, "rb" ) )

    [x, y, pred]  = [ test_result['xgold'], test_result['ygold'], test_result['pred']]
    print(lang_codes)
    tmpdf = pd.DataFrame(y)
    tmpdf['pred']= pred
    print(tmpdf['label'])
    indexes = tmpdf['label'].isin(lang_codes)
    tmpdf = tmpdf[indexes].reindex()
    print(tmpdf['label'])
    y = tmpdf['label']
    pred = tmpdf['pred']
    pred = [preprocess_pred(p, labels) for p in pred]
    return [x, y, pred]

def preprocess_pred(in_x, labels):
    in_x = in_x.upper()

    if in_x.upper() == 'MSA':
        in_x = 'MYS'

    if in_x.upper() == 'SLK':
        in_x = 'SVK'

    if in_x.upper() == 'SRP':
        in_x = 'SRB'

    if in_x.upper() == 'POR':
        in_x = 'PRT'

    if in_x.upper() == 'CES':
        in_x = 'CZE'

    if in_x.upper() == 'BUL':
        in_x = 'BGR'

    if in_x.upper() == 'SPA':
        in_x = 'ESP'

    if in_x.upper() == 'CAT':
        in_x = 'ESP'

    if in_x.upper() in ['ENG']:
        in_x = 'MOZ'

    x=in_x

    country = pycountry.countries.get(alpha3=x.upper())
    pred = country.alpha2.lower()
    if pred == 'in':
        pred = 'id'

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
    lang_codes= ['bg','es-ES','my','sr','hr','pt-BR','cz','es','id','pt-PT','es-AR','mk','sk','xx', 'pt', 'cs']

    goldData = read_data_filtered_by_lang_codes('../data/test/test-gold.txt', lang_codes)

    lang_codes= ['bg','es-ES','my','sr','hr','pt-BR','cz','es','id','pt-PT','es-AR','mk','sk','xx', 'pt', 'cs']
    labels = ['bg', 'cz','es', 'esAR', 'esES', 'hr', 'id', 'mk', 'my', 'pt', 'ptBR', 'ptPT', 'sk', 'sr', 'xx']
    [Xgold, ygold, pred] = load_results('output/benchmarks/rosette.dat', labels, lang_codes)

    print ('test', set(pred))
    ygold = [preprocess_gold(g) for g in ygold.values]
    [cm, cr] = get_confusion_matrix(pred, ygold)
    labels = ['bg', 'cz','es', 'hr', 'id', 'mk', 'my', 'pt', 'sk', 'sr', 'xx']
    save_cm(cm, 'output/benchmarks/rosette_cm.png', labels)

if __name__=='__main__':
    main()
