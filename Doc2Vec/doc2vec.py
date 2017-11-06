# -*- coding:utf-8 -*-
import os
import sys
import pickle
import argparse
import logging
from random import shuffle
from convert_to_sentences import convert_to_sentences
from time import gmtime, strftime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_classifier.sentence_classifier import SentenceClassifier

# Set argparse.
parser = argparse.ArgumentParser(description='Convert sentences and paragraphs into a dense representation.')

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('logs/doc2vec.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Set sentence classifier.
sent_cls_path = 'sentence_classifier/classifiers/17-11-03_18:45/sentence_classifier_17-11-03_18:45.pkl'
sent_cls_names_path = 'sentence_classifier/classifiers/17-11-03_18:45/sentence_classifier_dict_17-11-03_18:45.pkl'
sent_cls = SentenceClassifier()
sent_cls.load_classifier(sent_cls_path)
sent_cls_names = pickle.load(open(sent_cls_names_path, 'r'))
sent_cls.set_names(sent_cls_names)


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []

    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            pred = sent_cls.predict_class(line)
            if pred[0]:
                yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])
            else:
                continue

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def get_model_path():

    logger.info('Generating output path.')
    if not os.path.isdir('models'):
        os.makedirs('models')

    return 'models/model_' + strftime("%Y-%m-%d_%H-%M-%S.doc2vec", gmtime())


def train_model(sentences, model=None):
    logger.info('Training model.')

    if not model:
        model = Doc2Vec(size=100, window=2, min_count=2, workers=2, alpha=0.025, min_alpha=0.025)

    model.build_vocab(sentences)

    for epoch in range(10):
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    output_path = get_model_path()

    logger.info('Saving trained model.')
    model.save(output_path)

    return output_path


def create_sent_dict(sentences):

    s_dict = dict()

    for sent in sentences:
        s_dict[sent[1][0]] = sent[0]

    return s_dict


if __name__ == "__main__":

    parser.add_argument('--train', type=str, help='Please, provide a valid path to a file containing text.')
    parser.add_argument('--preprocess', type=bool, help='Indicate if the text in the file needs preprocessing'
                        '(True or False).', default=False)
    parser.add_argument('--test', type=str, help='Provide a valid path to a doc2vec model.')
    parser.add_argument('--model', type=str, help='Please, provide a valid path to a file containing a doc2vec model.')

    args = parser.parse_args()

    if args.train:

        file_path = args.train

        # Get sentences.
        if args.preprocess:
            file_path = convert_to_sentences(file_path)

        sentences = LabeledLineSentence(file_path)

        # Create a dict to convert a sent code into its respective sentence.
        sent_dict = create_sent_dict(sentences)

        if not args.model:
            output_model = train_model(sentences)
        else:
            old_model = Doc2Vec.load(args.model)
            output_model = train_model(sentences, old_model)

        base, _ = os.path.splitext(output_model)

        # Save the dict.
        pickle.dump(sent_dict, open(base + '.pkl', 'w'))

    elif args.test:
        model = Doc2Vec.load(args.test)
        # print model.docvecs.most_similar(20)
        print model.infer_vector('This shall be respected.')

    else:
        print "Nothing to do here. Use python doc2vec.py -h"
