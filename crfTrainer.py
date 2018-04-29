from model import Model
from sklearn.externals import joblib
import sklearn_crfsuite
import scipy
import sys
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.grid_search import RandomizedSearchCV
from crfUtil import gen_test_train_files, loadCorpus, view_issues

sys.path.insert(0, '.')


def gen_model(train, test, writeFile=False):
    model = Model()
    X_train = [model.sentence2features(s) for s in train]
    y_train = [model.sentence2labels(s) for s in train]

    X_test = [model.sentence2features(s) for s in test]
    y_test = [model.sentence2labels(s) for s in test]

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

    if writeFile:
        joblib.dump(crf, 'model.pkl')

    return metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    )


if __name__ == '__main__':
    gen_test_train_files('corp.tsv')
    train = loadCorpus('train.tsv')
    test = loadCorpus('test.tsv')

    print(gen_model(train, test, True))

    view_issues()
