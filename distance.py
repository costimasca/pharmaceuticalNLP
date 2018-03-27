import nltk, os
from sklearn.externals import joblib
from apted import helpers,apted, Config
import subprocess
import re
from model import getLabels

clf = joblib.load('model.pkl')


def convert_to_IOB(sent):
    new_sent = []
    inside_entity = {'O':False,'DOS':False,'UNIT':False,'WHO':False, 'FREQ':False, 'PER':False}

    for i,(t,p,n) in enumerate(sent):
        if inside_entity[n]:
            new_sent.append((t, p, 'I-' + n))
        else:
            new_sent.append((t, p, 'B-' + n))
            inside_entity = dict.fromkeys(inside_entity,False)
            inside_entity[n] = True

    return new_sent


def convert_bracket_notation(sent):
    """Converts from nltk paranthesis notation to bracket notation

    (A(B(X)(Y)(F))(C)) -> {A{B{X}{Y}{F}}{C}}
    """

    sent = sent.replace('(','{')
    sent = sent.replace(')','}')
    sent = sent.replace('{/{','(/(')
    sent = sent.replace('}/}',')/)')

    return sent


def convert_to_ne_text_tree(sentence):
    """takes a string sentence as input and returns a tree structure
    with named entities grouped in subtrees

    returns a bracket notation tree
    """

    ne_labels = getLabels(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O', 'FREQ', 'PER'])
    tree = convert_bracket_notation(tree.__str__())

    return tree


def convert_to_ne_tree(sentence):
    """takes a string sentence as input and returns a tree structure
    with named entities grouped in subtrees

    returns a bracket notation tree
    """

    ne_labels = getLabels(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O', 'FREQ', 'PER'])
    return tree


def view_tree(sentence):
    ne_labels = getLabels(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O', 'FREQ', 'PER'])
    tree.draw()

def calculate_distance_from_text_trees(text_tree1, text_tree2):
    tree1 = helpers.Tree('tree1').from_text(text_tree1)
    tree2 = helpers.Tree('tree2').from_text(text_tree2)

    class CustomConfig(Config):
        def rename(self, node1, node2):
            return int(node1.name.split(' ')[0] != node2.name.split(' ')[0])

        def children(self, node):
            return getattr(node, 'children', [])

    a = apted.APTED(tree1, tree2, CustomConfig())
    ted = a.compute_edit_distance()
    return ted


def generate_distance_matrix(folder):
    """Given a folder that contains files with single sentences,
     returns the TED matrix
    """
    files = sorted(os.listdir(folder))

    sentences = []
    for f in files:
        with open(folder + f) as drug:
            sentences.append(convert_to_ne_tree(drug.readlines()[0]))
            drug.close()

    output = open('matrix.csv','w')
    for s1 in sentences:
        row = [s1]
        for s2 in sentences:
            row.append(str(calculate_distance_from_text_trees(s1,s2)))
        output.write(','.join(row) + '\n')

    output.close()


def all_structures(folder='./train_full/'):
    files = sorted(os.listdir(folder))

    structures = {}
    sentences = []
    for f in files:
        with open(folder + f) as drug:
            sentences.append((drug.readline(),f))
            drug.close()

    for sent,file in sentences:
        tree = convert_to_ne_tree(sent)
        subtrees = list(tree.subtrees())[1:]
        struct = ''
        for t in subtrees:
            struct += t.label() + ' '

        if struct in structures:
            structures[struct].append(file)
        else:
            structures[struct] = [file]

    for k,v in sorted(structures.items(),key=lambda x: len(x[1]),reverse=True):
        print(k + ":" + str(v))

    print(len(structures.items()))


if __name__ == '__main__':

    all_structures()

    first_struct = ['actoplus-met', 'adasuve', 'adreview', 'agrylin', 'alesse', 'alimta', 'alosetron-hydrochloride', 'alphagan-p', 'anoro-ellipta', 'apidra', 'aubagio', 'auvi-q', 'azopt', 'bactrim', 'bayrab', 'bendeka', 'betimol', 'bevespi-aerosphere', 'bidil', 'biltricide', 'cephadyn', 'cetacaine', 'clinolipid', 'colchicine', 'combigan', 'combunox', 'corlopam', 'covera-hs', 'crixivan', 'cycloset', 'dantrium', 'deconsal', 'deconsal-ct', 'dht', 'donnatal', 'duagen', 'duoneb', 'dynacirc-cr', 'dyrenium', 'eldepryl', 'elitek', 'ella', 'empliciti', 'enablex', 'entyvio', 'epipen', 'eraxis', 'estrostep', 'estrostep-fe', 'etopophos', 'eulexin', 'exalgo', 'faslodex', 'ferriprox', 'firazyr', 'flonase', 'floxuridine', 'fluress', 'fml', 'frova', 'genoptic', 'giazo', 'gilenya', 'gliadel', 'glyset', 'gonal-f', 'heparin-sodium-injection', 'hycamtin-capsules', 'hyperrab', 'imodium', 'incruse-ellipta', 'infergen', 'inspra', 'inversine', 'invokamet-xr', 'iplex', 'janumet', 'janumet-xr', 'jentadueto-xr', 'jevtana', 'junel-fe', 'keflex', 'kepivance', 'kerlone', 'konyne', 'lartruvo', 'latisse', 'lexiscan', 'liletta', 'lotemax', 'lotrel', 'lutera', 'lymphazurin', 'medrol', 'mepron', 'methylene-blue', 'miacalcin', 'micardis', 'mono-vacc', 'mugard', 'myfortic', 'nasonex', 'neotect', 'nexavar', 'nipent', 'nitro-dur', 'nitropress', 'novolin-70-30-innolet', 'novolin-n-innolet', 'nuplazid', 'omniscan', 'onglyza', 'onmel', 'opticrom', 'panhematin', 'pentasa', 'pentoxifylline', 'perforomist', 'perjeta', 'permax', 'praluent', 'pred-g', 'prevacid', 'primaquine', 'pristiq', 'procardia-xl', 'profilnine', 'provera', 'ranexa', 'relpax', 'revatio', 'rexulti', 'rowasa', 'seasonique', 'secreflo', 'siliq', 'simbrinza', 'skelid', 'skyla', 'sodium-lactate', 'solaraze', 'solesta', 'stiolto-respimat', 'stivarga', 'striverdi-respimat', 'stromectol', 'sulfamylon', 'sumavel-dosepro', 'synarel', 'tekamlo', 'tenivac', 'terazol', 'testoderm', 'thrombin', 'tice', 'tigecycline-generic', 'toviaz', 'trobicin', 'trusopt', 'tussigon', 'twynsta', 'ultracet', 'viadur', 'victrelis', 'viramune', 'visken', 'voraxaze', 'votrient', 'xadago', 'xalatan', 'xtandi', 'yellow-fever-vaccine', 'yervoy', 'zadaxin', 'zingo', 'zirgan', 'zontivity', 'zyflo']

    for drug in first_struct:
        with open('./train_full/'+drug) as f:
            sent = f.readline()
            f.close()
        print(view_tree(sent))

    # with open('./dosages/' + 'trezix') as f:
    #     sent = f.readline()
    #     f.close()
    # view_tree(sent)