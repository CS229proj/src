import sys
import os
import Slide as lid
import Utils
import evaluate
import pickle as pc

def main():
    train_file = "../../data/train/train.txt"
    test_file = "../../data/test/test.txt"
    gold_file = "../../data/test/test-gold.txt"

    slide = lid.Slide(max_ngram = 5)
    slide.train(train_file)
    pc.dump(slide, 'slide_trained.dat')
    predictions = slide.predict(test_file)

    gold_labels = Utils.get_y(gold_file)
    evaluate.breakdown_evaluation(predictions, gold_labels)

if __name__=='__main__':
    main()