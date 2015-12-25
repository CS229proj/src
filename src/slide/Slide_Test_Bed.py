import sys
import os
import Slide as lid
import Utils
import evaluate
import pickle as pc
import matplotlib.pyplot as plt
import pandas as pd

def main(train_file, test_file, gold_file):

    slide = lid.Slide()
    #slide.train(train_file)
    #slide.load_label_encoder(train_file)
    slide.load_model('saved_models/slide_trained.dat')
    #slide.save_model('saved_models/slide_trained.dat')

    predictor_list = [0,1,2,3,3]

    test_data = pd.read_csv(test_file, encoding='utf-8', sep=r'\t+', header=None, names=['text'])
    X_test_raw = test_data['text'].values
    predictions = slide.predict(X_test_raw, predictor_list)

    gold_labels = Utils.get_y(gold_file)
    evaluate.breakdown_evaluation(predictions, gold_labels)


def parameter_testing(train_file, test_file, gold_file):

    slide = lid.Slide()
    slide.load_model('saved_models/slide_trained.dat')

    test_data = pd.read_csv(test_file, encoding='utf-8', sep=r'\t+', header=None, names=['text'])
    X_test_raw = test_data['text'].values
    gold_labels = Utils.get_y(gold_file)

    param_num = 4
    accuracy_list = []
    for i0 in xrange(param_num):
        predictor_list0 = [i0]
        overall_accuracy = calculate_accuracy(X_test_raw, gold_labels, slide, predictor_list0, False, True)
        accuracy_list.append((predictor_list0, overall_accuracy))
        for i1 in [ x for x in xrange(param_num) if x >= i0]:
            predictor_list1 = predictor_list0
            predictor_list1.append(i1)
            overall_accuracy = calculate_accuracy(X_test_raw, gold_labels, slide, predictor_list1, False, True)
            accuracy_list.append((predictor_list1, overall_accuracy))

            for i2 in [ x for x in xrange(param_num) if x >= i1]:
                predictor_list2 = predictor_list1
                predictor_list2.append(i2)
                overall_accuracy = calculate_accuracy(X_test_raw, gold_labels, slide, predictor_list2, False, True)
                accuracy_list.append((predictor_list2, overall_accuracy))

                for i3 in [ x for x in xrange(param_num) if x >= i2]:
                    predictor_list3 = predictor_list2
                    predictor_list3.append(i3)
                    overall_accuracy = calculate_accuracy(X_test_raw, gold_labels, slide, predictor_list3, False, True)
                    accuracy_list.append((predictor_list3, overall_accuracy))

                    for i4 in [ x for x in xrange(param_num) if x >= i3]:
                        predictor_list4 = predictor_list4
                        predictor_list4.append(i4)
                        overall_accuracy = calculate_accuracy(X_test_raw, gold_labels, slide, predictor_list4, False, True)
                        accuracy_list.append((predictor_list4, overall_accuracy))

    plot_accuracy_list(accuracy_list)


def plot_accuracy_list(accuracy_list):
    print('plot_accuracy_list')
    x = xrange(len(accuracy_list))
    y = [tupple[1] for tupple in accuracy_list]

    labels = [str(item) for item in [tupple[0] for tupple in accuracy_list]]

    print('labels', labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0,1)
    plt.plot(x,y, 'ro')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    for i,j in zip(x,y):
        ax.annotate(str(i)+','+str(j),xy=(i,j))

    plt.ylabel('Accuracy')
    plt.savefig('plot_accuracy_list.png')
    print('plot_accuracy_list done!')


def calculate_accuracy(test_file, gold_labels, slide, predictor_list, human_readable, overall_only):
    predictions = slide.predict(test_file, predictor_list)
    overall_accuracy = evaluate.breakdown_evaluation(predictions, gold_labels, human_readable, overall_only)
    print('overall_accuracy', overall_accuracy, 'predictor_list', predictor_list)
    return overall_accuracy



if __name__=='__main__':

    train_file = "../../data/train/train.txt"
    test_file = "../../data/test/test.txt"
    gold_file = "../../data/test/test-gold.txt"

    parameter_testing(train_file, test_file, gold_file)