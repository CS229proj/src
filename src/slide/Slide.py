import operator
import pandas as pd
from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import *
from itertools import izip, chain,islice
from sklearn import preprocessing
import CharTokenize as ct
import Utils
import pickle as pc
from passage.utils import save, load
import os
class Slide(object):


    def __init__(self):
        self.__label_encoder = preprocessing.LabelEncoder()
        self.__trained_models = {}


    def __get_models(self, model_repository_path):
        models = {}
        for key in os.listdir(model_repository_path):
            absolute_path = os.path.join(model_repository_path, key)
            if os.path.isdir(absolute_path):
                models[key] = absolute_path

        return models

    def __fit_model(self, X, Y, num_features):

        layers = [
            Embedding(size=128, n_features=num_features),
            GatedRecurrent(size=512, p_drop=0.4),
            Dense(size=14, activation='softmax', p_drop=0.2)
        ]

        model = RNN(layers=layers, cost='cce', updater=Adadelta(lr=0.5))
        model.fit(X, Y, n_epochs=10)

        return model

    def train(self, train_file):
        print('training ', train_file)
        train_data = pd.read_csv(train_file, encoding='utf-8', sep=r'\t+', header=None, names=['text', 'label'])

        X_train_raw = train_data['text'].values
        Y_train_raw = train_data['label'].values

        print(X_train_raw.shape)
        print(Y_train_raw.shape)

        for num_ngram in [
            #4,3,2,
            #5,
            1
        ]:

            print('current_ngram : ', num_ngram)
            tokenizer = ct.CharTokenize(character=True, charn=num_ngram, min_df=2, max_features=1000000)
            #tokenizer = ct.CharTokenize(min_df=2, max_features=1000000)

            X_train_vectorized = Utils.create_term_document_matrix(X_train_raw, tokenizer)
            Y_train_vectorized = Utils.vectorize_y(Y_train_raw, self.__label_encoder)

            print(len(X_train_vectorized))
            print(len(Y_train_vectorized))
            print(tokenizer.num_features)

            model = self.__fit_model(X_train_vectorized, Y_train_vectorized, tokenizer.num_features)
            self.__trained_models.append((tokenizer, model))

    def load_label_encoder(self, train_file):
        print('load_label_encoder')
        train_data = pd.read_csv(train_file, encoding='utf-8', sep=r'\t+', header=None, names=['text', 'label'])
        Y_train_raw = train_data['label'].values
        Utils.vectorize_y(Y_train_raw, self.__label_encoder)

    def __load(self, tmp_path):
        print('loading ', tmp_path)
        tmp_file = open(tmp_path, 'rb')
        tmp_obj = pc.load(tmp_file)
        tmp_file.close()
        print('loaded ', tmp_path)
        return tmp_obj

    def predict_batch(self, X_test_raw, predictor_list):

        preds = []

        if(len(predictor_list) > 1):
            selected_models = operator.itemgetter(*predictor_list)(self.__trained_models)
        else:
            selected_models = [self.__trained_models[0]]

        for (tokenizer, model) in selected_models:
            X_test_vectorized = Utils.create_document_term_matrix(X_test_raw, tokenizer)
            predictions = model.predict(X_test_vectorized)
            Y_test_predicted_vectorized = np.argmax(predictions, axis=1)

            Y_test_raw = Utils.devectorize_y(Y_test_predicted_vectorized, self.__label_encoder)
            preds.append(Y_test_raw)

        preds = zip(*preds)
        Y_test_predicted = map(lambda x: Utils.most_common(x), preds)

        return Y_test_predicted

    def predict(self, query_text, predictor_list):

        preds = []

        selected_models = []

        print(predictor_list)

        for key in predictor_list:
            print(key)
            if key in self.__trained_models.keys():
                tmp_model = self.__trained_models[key]
                selected_models.append(tmp_model)

        print(selected_models)

        for (tokenizer, model) in selected_models:
            print(tokenizer)
            print(model)
            query_vectorized = Utils.create_document_term_matrix(query_text, tokenizer)
            predictions = model.predict(query_vectorized)
            prediction_vectorized = np.argmax(predictions, axis=1)
            prediction_raw = Utils.devectorize_y(prediction_vectorized, self.__label_encoder)
            preds.append(prediction_raw)

        if len(preds) == 0:
            return 'xx'

        preds = zip(*preds)
        print('preds', preds)
        prediction = Utils.most_common(preds)
        return prediction


    def save_model(self, filename):
        import sys
        sys.setrecursionlimit(100000)
        i = 1
        for (tokenizer, model) in self.__trained_models:
            try:
                print('saving model', i)
                tmp_model_filename =  filename + '.'+ str(i) + '.model'
                model_file = open(tmp_model_filename, 'wb')
                pc.dump(model, model_file, 2)
                model_file.close()
                print('model saved', i)

                print('saving tokenizer', i)
                tmp_tokenizer_filename =  filename + '.'+ str(i) + '.tokenizer'
                tokenizer_file = open(tmp_tokenizer_filename, 'wb')
                pc.dump(tokenizer, tokenizer_file, 2)
                tokenizer_file.close()
                print('tokenizer saved', i)
            except Exception, e:
                print('Saving ERROR !!!', i, str(e), 'will continue with saving next model')
            i = i + 1

        print('saving encoder')
        tmp_encoder_filename =  filename + '.encoder'
        encoder_file = open(tmp_encoder_filename, 'wb')
        pc.dump(self.__label_encoder, encoder_file, 2)
        encoder_file.close()
        print('encoder saved')

        print('model saved')

    def load_model(self, model_repository_path):

        models = self.__get_models(model_repository_path)
        print('models', models)
        self.__trained_models = {}

        for key, model_path in models.iteritems():
            try:
                model = self.__load(os.path.join(model_path, 'slide.model'))
                tokenizer = self.__load(os.path.join(model_path, 'slide.tokenizer'))
                self.__trained_models[key] = (tokenizer, model)
            except Exception, e:
                print('Exception while loading ', model_path, str(e))


        print('loading encoder')
        self.__label_encoder = self.__load(os.path.join(model_repository_path, 'slide.encoder'))
        print('encoder loaded')

        print('1-len(self.__trained_models):', len(self.__trained_models))
