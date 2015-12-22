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

    slide = lid.Slide()
    slide.train(train_file)
    output = open('slide_trained.dat', 'wb')
    pc.dump(slide, output, -1)
    output.close()
    predictions = slide.predict(test_file)

    gold_labels = Utils.get_y(gold_file)
    evaluate.breakdown_evaluation(predictions, gold_labels)

if __name__=='__main__':
    main()