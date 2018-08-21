import os
import sys
import matplotlib.pyplot as plt
import nltk
import csv
from scipy.stats import norm
import matplotlib.mlab as mlab
from model.crf_model import Model
from model.crf_trainer import Trainer

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


def lists_equal(l1,l2):
    if len(l1) != len(l2):
        return False

    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False

    return True


def view_issues(file='model/corp.tsv'):
    """generates a tsv file with labels given to test sentences.
    the purpose of this is to see where most errors occur.
    perhaps a clue to a correction in the data set can be inferred from this"""
    corp = loadCorpus(file)
    model = Model('model.pkl')

    entities = ['B-DOS', 'O-DOS', 'B-DUR', 'I-DUR', 'O-DUR', 'B-FREQ',
                'I-FREQ', 'O-FREQ', 'B-UNIT', 'I-UNIT', 'O-UNIT', 'B-WHO', 'I-WHO', 'O-WHO']

    for sent in corp:
        new_sent = ' '.join([el[0] for el in sent])
        labels = [el[2] for el in sent]

        prediction = model.predict(new_sent)
        print(prediction)
        if prediction:
            pred = []
            for p in prediction:
                pred += p
            prediction = [p[1] for p in pred]

        if not lists_equal(labels, prediction):
            j = corp.index(sent)
            for i in range(len(sent)):
                c = corp[j][i][2]
                p = prediction[i]

                if c != p:
                    for en in entities:
                        if corp[j][i][2] == en or prediction[i] == en:
                            corp[j][i][2] += ' !!! ' + prediction[i]

    writeTsv(corp, 'issues.tsv')


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

    for i, sent in enumerate(sentences):
        sentences[i] = sent[:-3] + '.'

    for sent in sentences:
        t.write(sent + '\n')


def fix_eg_ie():
    c = open('corpus.tsv')
    lines = c.readlines()

    t = open('corp2.tsv', 'w')
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
    model = Model()
    corp = loadCorpus('corp2.tsv')

    for sent in corp:
        new_sent = ' '.join([el[0] for el in sent])
        labels = [el[2] for el in sent]

        prediction = model.predict(new_sent)
        if prediction:
            pred = []
            for p in prediction:
                pred += p
            prediction = [p[1] for p in pred]

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

   n, bins, patches = plt.hist(p, normed=1, facecolor='grey', alpha=0.75)

   y = mlab.normpdf(bins, mu, sigma)
   l = plt.plot(bins, y, 'b--', linewidth=2)

   # plt.title("f1 error distribution")
   plt.grid(False)

   plt.show()


def fix_slashes_dashes_on_sent_file():
    f = open('all_sentences')
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

    f = open('corpus2_split', 'w')
    for sent in new_sentences:
        f.write(sent + '\n')

    f.close()


def sent_file_to_unlab_corp(file):
    """
    Generates a unlabeled corpus file.
    :param file: File containing sentences that were preprocessed, one per line.
    :return: None
    """
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


def change_format():
    file = 'model/test.tsv'
    with open(file) as corpus_file:
        corp = corpus_file.readlines()

    new_corp = []

    for i, line in enumerate(corp):
        if line.endswith('\tI-WHO\n'):
            if not corp[i+1].endswith('\tI-WHO\n'):
                line = line[:-7] + '\tO-WHO\n'
        new_corp.append(line)

    for sent in new_corp:
        print(sent)

    with open('model/tmp.tsv', 'w') as f:
        for line in new_corp:
            f.write(line)


def distrib():
    file = 'model/corp.tsv'
    with open(file) as corpus_file:
        corp = corpus_file.readlines()

    count = {'DOS': 0, 'UNIT': 0, 'DUR': 0, 'FREQ': 0, 'WHO': 0}

    for line in corp:
        label = line.split('-')[-1][:-1]
        if label in count.keys():
            count[label] += 1

    count = {key: c/1401 for key, c in count.items()}
    print(count)


def average_results():
    file = 'model/results'
    with open(file) as f:
        results = f.readlines()

    new_list = []
    tmp = []
    for line in results:
        if line == '\n':
            new_list.append(tmp)
            tmp = []
        else:
            line = line.split(',')[:-1]
            tmp.append(line)

    result_dict = {}
    for line in new_list[0]:
        result_dict.update({line[0]: []})

    for res in new_list:
        for line in res:
            result_dict[line[0]].append(line[1:])

    print(result_dict.keys())

    for entity, result_list in result_dict.items():
        precision = 0
        recall = 0
        f1 = 0

        for res in result_list:
            precision += float(res[0])
            recall += float(res[1])
            f1 += float(res[2])

        precision /= 6
        recall /= 6
        f1 /= 6
        print(entity + ' & %.3f & %.3f & %.3f \\\\' % (precision, recall, f1))


if __name__ == '__main__':
    # change_format()

    average_results()
    # trainer = Trainer('model.pkl')
    # trainer.validate_performance('model/test.tsv')
    # trainer.generate_model('model/corp.tsv')
    # trainer.generate_model('model/corp.tsv')
    # view_issues()
