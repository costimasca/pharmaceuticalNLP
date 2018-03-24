import nltk, os
from sklearn.externals import joblib
from crfUtil import *
from apted import helpers,apted, Config
import subprocess
import re

clf = joblib.load('model.pkl')


def get_model_tags(sentence):
    sentence = nltk.word_tokenize(sentence)
    sent = nltk.pos_tag(sentence)
    sent = sent2features(sent)

    labels = clf.predict([sent])
    return labels


def convert_to_IOB(sent):
    new_sent = []
    inside_entity = {'O':False,'DOS':False,'UNIT':False,'WHO':False}

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


def convert_to_ne_tree(sentence):
    """takes a string sentence as input and returns a nltk.tree structure
    with named entities grouped in subtrees

    returns a bracket notation tree
    """

    ne_labels = get_model_tags(sentence)[0]
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O'])
    tree = convert_bracket_notation(tree.__str__())

    return tree


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

if __name__ == '__main__':
    sent1 = 'The recommended dosage for Xanax is 10 mg each day for adult patients.'
    sent2 = 'The recommended dosage for Xanax is 10 mg each day for adult patients.'

    sent1 = convert_to_ne_tree(sent1)
    sent2 = convert_to_ne_tree(sent2)

    dist = calculate_distance_from_text_trees(sent1, sent2)

    print(dist)

