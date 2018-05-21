from model.crf_model import Model
import sklearn_crfsuite
import scipy
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
import random
import csv
from sklearn.externals import joblib
import argparse


class Trainer:
    def __init__(self):
        self.model = Model()

    def generate_model(self, data_set):
        X_train, y_train, X_test, y_test = self.__gen_test_train__(data_set)

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

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
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)
        rs.fit(X_train, y_train)

        crf = rs.best_estimator_

        y_pred = crf.predict(X_test)

        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        joblib.dump(crf, 'model.pkl')

        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3))

        return metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        )

    def __gen_test_train__(self, corpus_file):
        """
        Given the corpus file, it will generate the train and test sets
            whose cardinality is in a 90%/10% ratio.
        """
        sentences = self.__loadCorpus__(corpus_file)

        test_number = int(len(sentences) * 0.1)
        test = random.sample(sentences, test_number)
        train = list(sent for sent in sentences if sent not in test)

        X_train = [self.model.sentence2features(s) for s in train]
        y_train = [self.model.sentence2labels(s) for s in train]

        X_test = [self.model.sentence2features(s) for s in test]
        y_test = [self.model.sentence2labels(s) for s in test]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def __write_tsv__(sentences, name):
        with open(name, 'w') as f:
            for sent in sentences:
                for line in sent:
                    f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
                f.write('\n')
            f.close()

    @staticmethod
    def __loadCorpus__(file):
        with open(file, 'r') as f:
            data = list(csv.reader(f, delimiter='\t'))

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
    trainer = Trainer()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_set')
    args = parser.parse_args()

    trainer.generate_model(args.data_set)
