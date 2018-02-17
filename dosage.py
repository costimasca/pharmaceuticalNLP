#!/usr/bin/env python3

from sklearn.externals import joblib
import nltk,sys

clf = joblib.load('model.pkl')


def getDosage(sentence):
    global clf

    sentence = nltk.word_tokenize(sentence)
    sent = nltk.pos_tag(sentence)
    sent = sent2features(sent)

    labels = clf.predict([sent])
    dosage = []
    unit = []
    for i in range(0,len(labels[0])):
        if(labels[0][i] == 'DOS'):
            dosage.append(sentence[i])
        if(labels[0][i] == 'UNIT'):
            unit.append(sentence[i])

    return tuple((dosage,unit))


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
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
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

if __name__ =='__main__':
    # print(getDosage(sys.argv[1]))
    print(getDosage("Active Duodenal Ulcer ??\" The recommended oral dosage for adults is 300 mg once daily at bedtime.?"))