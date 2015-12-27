import sys
import os
import Slide as lid
import Utils
import evaluate
import pickle as pc
import matplotlib.pyplot as plt
import pandas as pd
import copy


def run(model_repository_path, query_text):
    slide = lid.Slide()
    slide.load_model(model_repository_path = 'saved_models')

    predictor_list = ['2G', '3G', '4G', '5G', 'WORDG']
    predictor_list = ['1G', '2G']

    prediction = slide.predict(query_text, predictor_list)

    print('query_text', query_text)
    print('prediction', prediction)



if __name__=='__main__':

    model_repository_path = "saved_models"
    query_text = sys.argv[1]
    run(model_repository_path)