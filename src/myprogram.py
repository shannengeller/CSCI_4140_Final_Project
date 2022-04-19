#!/usr/bin/env python
from encodings import utf_8
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from corpus import english, French, russian 

class MyModel:

    @classmethod
    def load_training_data(cls):

        # opening english text file
        english_text = "autumn-leaves"
        raw_text_english = open(english_text + ".txt", "r", encoding= 'utf-8').read()
        raw_text_lowercase_english = raw_text_english.lower()

        # opening french text file
        french_text = "Germain.txt"
        raw_text_french = open(french_text + ".txt", "r", encoding= 'utf-8').read()
        raw_text_lowercase_french = raw_text_english.lower()

        #cleaning the english text (getting rid of numbers)
        clean_english_text = ' '.join(c for c in raw_text_lowercase_english if not c.isdigit()) 

        #cleaning the french text (getting rid of numbers)
        clean_french_text = ' '.join(c for c in raw_text_lowercase_french if not c.isdigit()) 

        #create a list of all characters within the english training text to see what the most common letters are
        all_eng_chars = sorted(list(set(clean_english_text)))

        #create a list of all characters within the french training text to see what the most common letters are
        all_fren_chars = sorted(list(set(clean_french_text)))
        
        #map a dictionary to hold letters and numerical values to them (ENGLISH)
        eng_chars_to_num = dict((c,i) for i,c in enumerate(all_eng_chars))

        #map a dictionary to hold letters and numerical values to them (FRENCH)
        fren_chars_to_num = dict((c,i) for i,c in enumerate(all_fren_chars))

        #reverse so we can correlate numbers to letters (ENGLISH)
        num_to_eng_chars = dict((i,c) for i,c in enumerate(all_eng_chars))

        #reverse so we can correlate numbers to letters (FRENCH)
        num_to_fren_chars = dict((i,c) for i,c in enumerate(all_fren_chars))

        #Showing all counts for english and french characters
        french_num_chars = len(clean_french_text)
        english_num_chars = len(clean_english_text)
        print("Number of characters in French text:", french_num_chars)
        print("Number of characters in English text:", english_num_chars)

        #creating input for training (ENGLISH text)
        seq_length = 60
        step = 50
        sentences = []
        next_chars = []

        for i in range(0, english_num_chars - seq_length, step):
            sentences.append(clean_english_text[i: i + seq_length])
            next_chars.append(clean_english_text[i + seq_length])

        n_patterns = len(sentences)
        print("Number of patterns:", n_patterns)

        #creating input for training (FRENCH text)
        seq_length = 60
        step = 50
        sentences_french = []
        next_chars_french = []

        for i in range(0, french_num_chars - seq_length, step):
            sentences_french.append(clean_french_text[i: i + seq_length])
            next_chars_french.append(clean_french_text[i + seq_length])
        
        n_patterns_french = len(sentences_french)
        print("Number of patterns:", n_patterns_french)

        

        return []

    @classmethod
    def load_test_data(cls, fname):
        # testing data given by instructor to see how model works 

        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # running our saved training model and trying to use given information and see what our best 3 choices wil be 

        pass

    def run_pred(self, data):
        # taking most common 3 letters using tensorflow function and most common words compared to what user inputs for test data

        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here

        # going to save whatever suggestions for the 3 letters to a text file for each line of astronauts text


        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
