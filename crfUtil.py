import os
import sys
import nltkTry

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.externals import joblib

nltk.corpus.conll2002.fileids()

import csv


# clf = joblib.load('model.pkl')


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


sys.path.insert(0, '.')


def makeCorpus():
    dir = 'DATASET/dosage/'
    l = sorted(os.listdir(dir))

    text = []

    for file in l:
        with open(dir + file, 'r') as f:
            text.append(nltkTry.parseString(f.read()))
            f.close()

    with open('corpus', 'w') as f:
        for sent in text:
            for tup in sent:
                f.write(tup[0] + '\t' + tup[1] + '\tO-DOSE\n')
            f.write('\n')
        f.close()


def remakeList(list):
    newList = []
    smallList = []

    for el in list:
        if el != ('',):
            smallList.append(el)
        else:
            newList.append(smallList)
            smallList = []

    # newList.append(smallList)

    return newList


def openFile(file):
    with open(file, 'r') as f:
        text = [tuple(line.strip().split('\t')) for line in f.readlines()]
        f.close()

    text = remakeList(text)

    return text


def writeIOB(corpus):
    newCorpus = []

    for sentence in corpus:
        newSentence = []
        for i in range(0, len(sentence)):
            token = sentence[i]

            if len(token) == 2:
                print(token)
                if i == 0:
                    token = token + ('B-DOSE',)
                elif len(sentence[i - 1]) == 3:
                    token = token + ('B-DOSE',)
                else:
                    token = token + ('I-DOSE',)
            newSentence.append(token)
        newCorpus.append(newSentence)

    return newCorpus


def getCorpus():
    file = '/home/constantin/Documents/practica/finalCorpus.tsv'

    return openFile(file)


def corpus2file(list, name):
    with open(name, 'w') as f:
        for line in list:
            for token in line:
                f.write(token[0] + '\t' + token[1] + '\t' + token[2] + '\n')
            f.write('\n')


def getDosage(sentence):
    global clf

    sentence = nltk.word_tokenize(sentence)
    sent = nltk.pos_tag(sentence)
    sent = sent2features(sent)

    labels = clf.predict([sent])
    dosage = []
    for i in range(0, len(labels[0])):
        if (labels[0][i] == 'DOS'):
            dosage.append(sentence[i])

    return dosage


def getAllDosages():
    dir = 'DATASET/dosage/'

    files = sorted(os.listdir(dir))
    text = []

    for file in files:
        with open(dir + file, 'r') as f:
            text.append(f.read())

    text2 = []

    for line in text:
        text2.append(getDosage(line))

    return text, text2


def makeDosageCorpus():
    file = 'dosageCorpus'
    corpus = 'dosage.tsv'

    with open(file, 'r') as f:
        text = f.readlines()
        f.close()

    text2 = []

    for line in text:
        sent = nltk.word_tokenize(line)
        sent = nltk.pos_tag(sent)
        text2.append(sent)

    with open(corpus, 'w') as f:
        for line in text2:
            for tup in line:
                f.write(tup[0] + '\t' + tup[1] + '\t' + 'O\n')
            f.write('\n')
        f.close()


def repairCorpus2():
    with open("corpus.tsv", r) as f:
        text = f.readlines()
        f.close()

    text2 = []
    sent = []
    for line in text:
        if line == '\t\t\n':
            text2.append(sent)
            sent = []
        else:
            sent.append(line.strip().split('\t'))

    return text2


def loadCorpus(file):
    with open(file, 'r') as f:
        l = list(csv.reader(f, delimiter='\t'))

    sentences = []
    sent = []

    for line in l:
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


def testTrain(sentences):
    import random
    test_number = int(len(sentences) * 0.2)
    test = random.sample(sentences, test_number)
    train = list(sent for sent in sentences if not sent in test)

    return train, test


def writeTsv(sentences, name):
    with open(name, 'w') as f:
        for sent in sentences:
            for line in sent:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
            f.write('\n')
        f.close()


def gen_test_train_files(corpus_file):
    """Given the .tsv corp file, it will generate 2 disjoint files, "train.tsv" and "test.tsv",
    whose cardinality is in a 80%/20% ratio. """
    sent = loadCorpus(corpus_file)
    train, test = testTrain(sent)
    writeTsv(train, "train.tsv")
    writeTsv(test, "test.tsv")


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def lists_equal(l1, l2):
    if len(l1) != len(l2):
        return False

    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False

    return True


def view_issues():
    """generates a tsv file with labels given to test sentences.
    the purpose of this is to see where most errors occur.
    perhaps a clue to a correction in the data set can be inferred from this"""
    test = loadCorpus('test.tsv')
    clf = joblib.load('model.pkl')

    corp = loadCorpus('corp.tsv')

    for sent in corp:
        new_sent = ' '.join([el[0] for el in sent])
        sentence = nltk.word_tokenize(new_sent)
        sentence = nltk.pos_tag(sentence)
        sentence = sent2features(sentence)

        labels = [el[2] for el in sent]

        prediction = clf.predict([sentence])[0]
        if not lists_equal(labels, prediction):
            j = corp.index(sent)
            for i in range(len(sent)):
                if corp[j][i][2] != prediction[i]:
                    corp[j][i][2] += '!!!' + prediction[i]

    writeTsv(corp, 'issues.tsv')


if __name__ == '__main__':
    view_issues()
