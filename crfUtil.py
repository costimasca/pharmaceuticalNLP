import os
import sys
import structure
import matplotlib.pyplot as plt
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
import csv
from scipy.stats import norm
import matplotlib.mlab as mlab
import model

nltk.corpus.conll2002.fileids()

# clf = joblib.load('model.pkl')

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


sys.path.insert(0, '.')


def corpus2file(list, name):
    with open(name, 'w') as f:
        for line in list:
            for token in line:
                f.write(token[0] + '\t' + token[1] + '\t' + token[2] + '\n')
            f.write('\n')

def getAllDosages():
    dir = 'DATASET/dosage/'

    files = sorted(os.listdir(dir))
    text = []

    for file in files:
        with open(dir + file, 'r') as f:
            text.append(f.read())

    text2 = []

    # for line in text:
    #     text2.append(getDosage(line))

    return text, text2


def makeCorpusFile():
    dir = './train_full/'
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    files = sorted(os.listdir(dir))
    text = []

    for file in files:
        with open(dir + file, 'r') as f:
            for line in f.readlines():
                for sent in sent_detector.tokenize(line):
                    if contains_named_entities(sent):
                        text.append(sent)
            f.close()

    with open('corpus2', 'w') as f:
        for line in text:
            f.write(line + '\n')
        f.close()


def makeDosageCorpus():
    file = 'corpus2_split'
    corpus = 'corp2.tsv'

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


def writeTsv(sentences, name):
    with open(name, 'w') as f:
        for sent in sentences:
            for line in sent:
                f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
            f.write('\n')
        f.close()


def gen_test_train_files(corpus_file='corp.tsv'):
    """Given the .tsv corp file, it will generate 2 disjoint files, "train.tsv" and "test.tsv",
    whose cardinality is in a 90%/10% ratio. """
    sentences = loadCorpus(corpus_file)

    import random
    test_number = int(len(sentences) * 0.1)
    test = random.sample(sentences, test_number)
    train = list(sent for sent in sentences if sent not in test)

    writeTsv(train, "train.tsv")
    writeTsv(test, "test.tsv")


def ispunctuation(word):
    if word in ['.', ',', '(', ')', '[', ']', '{', '}', ':', ';']:
        return True
    return False


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
        'word.isdigit()': word.isdigit() or postag == 'CD',
        'word.ispunctuation()': ispunctuation(word),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.isdigit()': word2.isdigit() or postag2 == 'CD',
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
        })

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit() or postag1 == 'CD',
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.isdigit()': word2.isdigit() or postag2 == 'CD',
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
        })

    if i < len(sent)-3:
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        features.update({
            '+3:word.lower()': word3.lower(),
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:word.isdigit()': word3.isdigit() or postag3 == 'CD',
            '+3:postag': postag3,
            '+3:postag[:2]': postag3[:2],
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit() or postag1 == 'CD',
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    if ispunctuation(word):
        features['bias'] = 0

    return features


def lists_equal(l1,l2):
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
    corp = loadCorpus('corp.tsv')
    clf = joblib.load('model.pkl')

    for sent in corp:
        new_sent = ' '.join([el[0] for el in sent])
        sentence = nltk.word_tokenize(new_sent)
        sentence = nltk.pos_tag(sentence)
        sentence = sent2features(sentence)
        labels = [el[2] for el in sent]

        prediction = clf.predict([sentence])[0]
        if not lists_equal(labels,prediction):
            j = corp.index(sent)
            for i in range(len(sent)):
                if corp[j][i][2] != prediction[i]:
                    if corp[j][i][2] == "WHO" or prediction[i] == "WHO":
                        corp[j][i][2] += ' !!! ' + prediction[i]
                    if corp[j][i][2] == "DOS" or prediction[i] == "DOS":
                        corp[j][i][2] += ' !!! ' + prediction[i]
                    if corp[j][i][2] == "UNIT" or prediction[i] == "UNIT":
                        corp[j][i][2] += ' !!! ' + prediction[i]
                    if corp[j][i][2] == "FREQ" or prediction[i] == "FREQ":
                        corp[j][i][2] += ' !!! ' + prediction[i]
                    if corp[j][i][2] == "PER" or prediction[i] == "PER":
                        corp[j][i][2] += ' !!! ' + prediction[i]

    writeTsv(corp, 'issues.tsv')


def contains_named_entities(sentence):
    pred = model.label(sentence)
    if 'DOS' in pred or 'UNIT' in pred:
        return True


def fix_dashes_slashes():
    c = open('corp2.tsv')
    lines = c.readlines()

    t = open('corpus2','w')
    sentences = []
    sent = ''
    for i, line in enumerate(lines):
        word = line.split('\t')[0]
        if word == '' or i == len(lines) -1:
            sentences.append(sent)
            sent = ''
        else:
            if '-' in word:
                print(word)
                tmp = word.split('-')
                word = ' - '.join(tmp)
                print(word)
            if '/' in word:
                print(word)
                tmp = word.split('/')
                word = ' / '.join(tmp)
                print(word)

            sent += word + ' '

    for i,sent in enumerate(sentences):
        sentences[i] = sent[:-3] + '.'

    for sent in sentences:
        t.write(sent + '\n')


def fix_eg_ie():
    c = open('corpus.tsv')
    lines = c.readlines()

    t = open('corp2.tsv','w')
    new_lines = []
    sent = ''
    for line in lines:
        line_split = line.split('\t')
        word = line_split[0]
        if word == 'eg.' or word == 'e.g.':
            print(word)
            word = 'e.g'
        if word == 'ie.' or word == 'i.e.':
            print(word)
            word = 'i.e'

        new_line = '\t'.join([word]+line_split[1:])
        new_lines.append(new_line)

    t.writelines(new_lines)


def label_corpus():
    clf = joblib.load('model.pkl')
    corp = loadCorpus('corp2.tsv')

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
                    corp[j][i][2] = prediction[i]

    writeTsv(corp, 'corpus2.tsv')


def plot_error_distrib(x):
   avg = sum(x)/len(x)
   p = [y - avg for y in x]

   (mu,sigma) = norm.fit(p)

   n, bins, patches = plt.hist(p, normed=1, facecolor='green', alpha=0.75)

   y = mlab.normpdf(bins, mu, sigma)
   l = plt.plot(bins,y,'r--', linewidth=2)

   plt.title("Distributia erorilor de recall")
   plt.grid(True)

   plt.show()


def fix_slashes_dashes_on_sent_file():
    f = open('corpus')
    sentences = f.readlines()
    f.close()
    new_sentences = []

    for sent in sentences:
        new_sent = ''

        sent_list = nltk.word_tokenize(sent)
        l = len(sent_list)
        for i, word in enumerate(sent_list):
            if '-' in word:
                print(word)
                word = ' - '.join(word.split('-'))
                print(word)

            if '–' in word:
                print(word)
                word = ' – '.join(word.split('–'))
                print(word)

            if '/' in word:
                word = ' / '.join(word.split('/'))

            if word == 'eg.' or word == 'e.g.':
                print(word)
                word = 'e.g'
            if word == 'ie.' or word == 'i.e.':
                print(word)
                word = 'i.e'

            new_sent += word
            if i != l:
                new_sent += ' '

        new_sent = new_sent.replace('  ', ' ')

        new_sentences.append(new_sent)

    f = open('corpus2_split','w')
    for sent in new_sentences:
        f.write(sent)

    f.close()


def sent_file_to_unlab_corp():
    file = 'corpus2_split'
    corpus = 'corpus.tsv'

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


def calc_performance(valid):
    """calculates the performance for a model on the given validation dataset"""
    test = loadCorpus(valid)

    X_test = [sent2features(s) for s in test]
    y_test = [sent2labels(s) for s in test]

    crf = joblib.load('model.pkl')
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    labels.remove('O')
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    return metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    )


if __name__ == '__main__':
    print()
    print(calc_performance('tmp_validation.tsv'))
