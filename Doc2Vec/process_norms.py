import pickle
import numpy as np
from doc2vec import *
from nltk.tokenize import sent_tokenize
from sentence_classifier.sentence_classifier import SentenceClassifier


class ProcessNorms:

    def __init__(self, doc2vec_path, sent_cls_path, sent_cls_names_path, contract_path=None):
        """
        Initialize class instances.
        :param contract_path: Path to a contract.
        :param doc2vec_path: Path to a trained doc2vec model.
        :param sent_cls_path: Path to a sentence classifier path. (Classifies whether a sentence is a norm or nor)
        """
        # Process contract text to extract sentences.
        self.contract_path = contract_path
        if self.contract_path:
            self.contract_text = open(self.contract_path, 'r').read()
            self.contract_sents = sent_tokenize(self.contract_text)
        # Set sentence classifier.
        self.sent_cls_path = sent_cls_path
        self.sent_cls = SentenceClassifier()
        self.sent_cls.load_classifier(self.sent_cls_path)
        self.sent_cls_names = pickle.load(open(sent_cls_names_path, 'r'))
        # Set Doc2Vec Model.
        self.doc2vec_path = doc2vec_path
        self.doc2vec_model = Doc2Vec.load(self.doc2vec_path)

    @staticmethod
    def _train_d2v(self, sentences=None):

        # Set sentences.
        if sentences:
            # LabeledLineSentence()
            pass

    def infer_new_vectors(self, text, sentence=False):
        # Still not needed.
        pass
