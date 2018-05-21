"""
Implements the training algorithm for the CRF model.
"""
import argparse
import random
import csv

import sklearn_crfsuite
import scipy

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn_crfsuite import metrics

from model.crf_model import Model


class Trainer:
    """
    Generates a CRF model given a data set of labeled words.
    """

    def __init__(self):
        self.model = Model()

    def generate_model(self, data_set):
        """
        Generates a CRF model given the data set.
        It saves the model to disk with the name 'model.pkl'.
        :param data_set: Path to the labeled data set.
        :return: Performance results.
        """
        x_train, y_train, x_test, y_test = self.gen_test_train(data_set)

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(x_train, y_train)

        labels = list(crf.classes_)
        labels.remove('O')

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=labels)

        # search
        rand_search = RandomizedSearchCV(crf, params_space,
                                         cv=3,
                                         verbose=1,
                                         n_jobs=-1,
                                         n_iter=50,
                                         scoring=f1_scorer)
        rand_search.fit(x_train, y_train)

        crf = rand_search.best_estimator_

        y_prediction = crf.predict(x_test)

        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        joblib.dump(crf, 'model.pkl')

        print(metrics.flat_classification_report(
            y_test, y_prediction, labels=sorted_labels, digits=3))

        return metrics.flat_classification_report(
            y_test, y_prediction, labels=sorted_labels, digits=3
        )

    def gen_test_train(self, corpus_file):
        """
        Given the corpus file, it will generate the train and test sets
            whose cardinality is in a 90%/10% ratio.
        """
        sentences = self.__load_corpus__(corpus_file)

        test_number = int(len(sentences) * 0.1)
        test = random.sample(sentences, test_number)
        train = list(sent for sent in sentences if sent not in test)

        x_train = [self.model.sentence2features(s) for s in train]
        y_train = [self.model.sentence2labels(s) for s in train]

        x_test = [self.model.sentence2features(s) for s in test]
        y_test = [self.model.sentence2labels(s) for s in test]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def __write_tsv__(sentences, name):
        with open(name, 'w') as file:
            for sent in sentences:
                for line in sent:
                    file.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
                file.write('\n')
            file.close()

    @staticmethod
    def __load_corpus__(corpus_file):
        with open(corpus_file, 'r') as file:
            data = list(csv.reader(file, delimiter='\t'))

        sentences = []
        sent = []

        for line in data:
            if line == [] or line[0] == '':
                sentences.append(sent)
                sent = []
            else:
                sent.append(line)

        sentences.append(sent)

        for sent in sentences:
            if sent and sent[0] == '0' and sent[1] == '0':
                sent[0] = '.'
                sent[1] = '.'

        return sentences


if __name__ == '__main__':
    TRAINER = Trainer()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('data_set')
    ARGS = PARSER.parse_args()

    TRAINER.generate_model(ARGS.data_set)
