
import pandas as pd
from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import *
from sklearn import preprocessing
import CharTokenize as ct
import Utils

class Slide(object):


    def __init__(self):
        self.__label_encoder = preprocessing.LabelEncoder()
        self.__trained_models = []

    def __fit_model(self, X, Y, num_features):

        layers = [
            Embedding(size=128, n_features=num_features),
            GatedRecurrent(size=512, p_drop=0.4),
            Dense(size=14, activation='softmax', p_drop=0.2)
        ]

        model = RNN(layers=layers, cost='cce', updater=Adadelta(lr=0.5))
        model.fit(X, Y, n_epochs=1)

        return model

    def train(self, train_file):
        print('training ', train_file)
        train_data = pd.read_csv(train_file, encoding='utf-8', sep=r'\t+', header=None, names=['text', 'label'])

        X_train_raw = train_data['text'].values
        Y_train_raw = train_data['label'].values

        print(X_train_raw.shape)
        print(Y_train_raw.shape)

        for num_ngram in [2]:#,3,2,5]:

            print('current_ngram : ', num_ngram)
            tokenizer = ct.CharTokenize(character=True, charn=num_ngram, min_df=2, max_features=1000000)

            X_train_vectorized = Utils.create_term_document_matrix(X_train_raw, tokenizer)
            Y_train_vectorized = Utils.vectorize_y(Y_train_raw, self.__label_encoder)

            print(len(X_train_vectorized))
            print(len(Y_train_vectorized))
            print(tokenizer.num_features)

            model = self.__fit_model(X_train_vectorized, Y_train_vectorized, tokenizer.num_features)
            self.__trained_models.append((tokenizer, model))

    def predict(self, test_file):
        print('predicting ', test_file)
        test_data = pd.read_csv(test_file, encoding='utf-8', sep=r'\t+', header=None, names=['text'])
        X_test_raw = test_data['text'].values

        preds = []

        for (tokenizer, model) in self.__trained_models:
            X_test_vectorized = Utils.create_document_term_matrix(X_test_raw, tokenizer)
            predictions = model.predict(X_test_vectorized)
            Y_test_predicted_vectorized = np.argmax(predictions, axis=1)

            Y_test_raw = Utils.devectorize_y(Y_test_predicted_vectorized, self.__label_encoder)
            preds.append(Y_test_raw)

        Y_test_predicted = map(lambda x: Utils.most_common(x), preds)

        return Y_test_predicted