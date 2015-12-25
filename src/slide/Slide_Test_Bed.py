import sys
import os
import Slide as lid
import Utils
import evaluate
import pickle as pc
import matplotlib.pyplot as plt

def main(train_file, test_file, gold_file):

    slide = lid.Slide()
    #slide.train(train_file)
    #slide.load_label_encoder(train_file)
    slide.load_model('saved_models/slide_trained.dat')
    #slide.save_model('saved_models/slide_trained.dat')

    predictor_list = [0,1,2,3,3]
    predictions = slide.predict(test_file, predictor_list)

    gold_labels = Utils.get_y(gold_file)
    evaluate.breakdown_evaluation(predictions, gold_labels)


def parameter_testing(train_file, test_file, gold_file):

    slide = lid.Slide()
    slide.load_model('saved_models/slide_trained.dat')

    param_num = 2
    accuracy_list = []
    for i0 in xrange(param_num):
        predictor_list0 = [i0]
        #overall_accuracy = calculate_accuracy(test_file, slide, predictor_list0, True)
        #accuracy_list.append((predictor_list0, overall_accuracy))
        for i1 in [ x for x in xrange(param_num) if x > i0]:
            predictor_list1 = predictor_list0
            predictor_list1.append(i1)
            overall_accuracy = calculate_accuracy(test_file, gold_file, slide, predictor_list1, True)
            accuracy_list.append((predictor_list1, overall_accuracy))

            for i2 in [ x for x in xrange(param_num) if x > i1]:
                predictor_list2 = predictor_list1
                predictor_list2.append(i2)
                overall_accuracy = calculate_accuracy(test_file, gold_file, slide, predictor_list2, True)
                accuracy_list.append((predictor_list2, overall_accuracy))

                for i3 in [ x for x in xrange(param_num) if x > i2]:
                    predictor_list3 = predictor_list2
                    predictor_list3.append(i3)
                    overall_accuracy = calculate_accuracy(test_file, gold_file, slide, predictor_list3, True)
                    accuracy_list.append((predictor_list3, overall_accuracy))

    plot_accuracy_list(accuracy_list)


def plot_accuracy_list(accuracy_list):
    print('plot_accuracy_list')
    x = xrange(len(accuracy_list))
    y = [tupple[1] for tupple in accuracy_list]
    plt.plot(x, y, 'ro')
    plt.ylabel('Accuracy')
    plt.savefig('plot_accuracy_list.png')
    print('plot_accuracy_list done!')

def calculate_accuracy(test_file, gold_file, slide, predictor_list, overall_only):
    print(test_file)
    print(gold_file)
    predictions = slide.predict(test_file, predictor_list)
    gold_labels = Utils.get_y(gold_file)
    overall_accuracy = evaluate.breakdown_evaluation(predictions, gold_labels, overall_only)
    return overall_accuracy



if __name__=='__main__':

    train_file = "../../data/train/train.txt"
    test_file = "../../data/test/test.txt"
    gold_file = "../../data/test/test-gold.txt"

    parameter_testing(train_file, test_file, gold_file)