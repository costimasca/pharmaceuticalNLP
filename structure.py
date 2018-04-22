import nltk, os
from sklearn.externals import joblib
from apted import helpers,apted, Config
import subprocess
import re
from model import label

clf = joblib.load('model.pkl')


def convert_to_IOB(sent):
    new_sent = []
    inside_entity = {'O': False, 'DOS': False, 'UNIT': False, 'WHO': False, 'FREQ': False, 'PER': False}

    for i,(t,p,n) in enumerate(sent):
        if inside_entity[n]:
            new_sent.append((t, p, 'I-' + n))
        else:
            new_sent.append((t, p, 'B-' + n))
            inside_entity = dict.fromkeys(inside_entity, False)
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

    ne_labels = label(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS', 'UNIT', 'WHO', 'O', 'FREQ', 'PER'])
    tree = convert_bracket_notation(tree.__str__())

    return tree


def convert_to_ne_tree(sentence):
    """takes a string sentence as input and returns a tree structure
    with named entities grouped in subtrees

    returns a bracket notation tree
    """

    ne_labels = label(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS', 'UNIT', 'WHO', 'O', 'FREQ', 'PER'])

    return tree


def view_tree(sentence):
    ne_labels = label(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos],[x[1] for x in pos],ne_labels))
    sent = convert_to_IOB(sent)

    text = ''
    for t, p, n in sent:
        text += t + ' ' + p + ' ' + n + '\n'

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS', 'UNIT', 'WHO', 'O', 'FREQ', 'PER'])
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


def contains_named_entities(sentence):
    pred = label(sentence)
    if not pred:
        return False
    if 'WHO' in pred[0] or 'DOS' in pred[0] or 'UNIT' in pred[0]:
        return True


def all_structures(folder='./dosages/'):
    files = sorted(os.listdir(folder))

    structures = {}
    sentences = []
    for f in files:
        with open(folder + f) as drug:
            sentences += [(d,f) for d in drug.readlines()]
            drug.close()

    k = 0
    for sent,file in sentences:
        tree = convert_to_ne_tree(sent)
        subtrees = list(tree.subtrees())[1:]
        struct = ''

        for i,t in enumerate(subtrees):
            if t.label() == 'O' and len(t.leaves()) == 1 and t.leaves()[0][0] == 'or':
                struct += t.label() + '_or '
                k += 1
                continue
            if t.label() == 'O' and len(t.leaves()) == 1 and t.leaves()[0][0].strip() == 'to':
                struct += t.label() + '_to '
                k += 1
                continue
            if t.label() == 'O' and len(t.leaves()) == 1 and t.leaves()[0][0] == '-':
                struct += t.label() + '_- '
                k += 1
                continue

            struct += t.label() + ' '

        struct2 = struct.split(' ')[:-1]
        if len(struct2) > 1:
            if struct2[0] == 'O':
                struct2 = struct2[1:]

            if struct2[-1] == 'O':
                struct2 = struct2[:-1]

        struct = ' '.join(struct2)

        struct = struct.replace('DOS O_or DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_or DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_or DOS ', 'DOSAGE ')
        struct = struct.replace('DOS O_or DOS ', 'DOSAGE ')

        struct = struct.replace('DOS O_to DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_to DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_to DOS ', 'DOSAGE ')
        struct = struct.replace('DOS O_to DOS ', 'DOSAGE ')

        struct = struct.replace('DOS O_- DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_- DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOS UNIT O_- DOS ', 'DOSAGE ')
        struct = struct.replace('DOS O_- DOS ', 'DOSAGE ')

        struct = struct.replace('O_or', 'O')
        struct = struct.replace('O_to', 'O')
        struct = struct.replace('O_-', 'O')

        struct = struct.replace('DOS ', 'DOSAGE ')
        struct = struct.replace('DOS UNIT', 'DOSAGE')
        struct = struct.replace('DOSAGE UNIT', 'DOSAGE')


        if struct in structures:
            structures[struct].append(file)
        else:
            structures[struct] = [file]

    for key,v in sorted(structures.items(),key=lambda x: len(x[1]),reverse=True):
        print(key + ":" + str(v))

    print(len(structures.items()))

    print(k)


def all_structures2():
    structures = {}
    file = 'corpus'

    with open(file) as f:
        sentences = f.readlines()

    for sent in sentences:
        try:
            struct = chunk_sentence(sent)
        except Exception:
            print("exception chunking")
            print(sent)

        try:
            struct = [st.label() for st in struct]
        except Exception:
            print('exception struct')
            print(struct)

        if len(struct) > 1:
            if struct[0] == 'O':
                struct = struct[1:]

            if struct[-1] == 'O':
                struct = struct[:-1]

        struct = ' '.join(struct)

        if struct in structures:
            structures[struct].append(sent)
        else:
            structures[struct] = [sent]

    for key, v in sorted(structures.items(), key=lambda x: len(x[1]), reverse=True):
        print(key + ": " + str(len(v)) + ' ' + str(v))

    print(len(structures))


def chunk_sentence(sentence):

    sentence = fix_dashes_slashes(sentence)
    ne_labels = label(sentence)
    sent = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(sent)

    sent = list(zip([x[0] for x in pos], ne_labels))
    # sent = convert_to_IOB(sent)

    for i,t in enumerate(sent):
        if t[0] == 'or':
            sent[i] = ('or',t[1]+'_or')
        if t[0] == 'to':
            sent[i] = ('to',t[1]+'_to')
        if t[0] == '-':
            sent[i] = ('-',t[1]+'_-')
        if t[0] == '/':
            sent[i] = ('/',t[1]+'_/')

    grammar = r"""
    DOS: {<DOS.*>+}
    UNIT: {<UNIT.*>+}
    FREQ: {<FREQ.*>+}
    PER: {<PER.*>+}
    WHO: {<WHO.*>+}
    O: {<O>+}
    O_or: {<O_or>}
    O_to: {<O_to>}
    O_-: {<O_->}
    O_/: {<O_/>}
    DOSAGE: {<DOS><UNIT>?<O_.*><DOS>?<UNIT>}
    DOSAGE: {<DOS><UNIT>}
    O: {<DOS>}
    O: {<UNIT>}
    """

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sent)

    for st in result.subtrees(lambda t: '_' in t.label()):
        st.set_label(st.label().split('_')[0])

    for leafPos in result.treepositions('leaves'):
        result[leafPos] = result[leafPos][0]

    res = nltk.Tree('S', [])
    i = 0
    while i < len(result):
        t = result[i]
        if t.label() == 'O':
            leaves = t.leaves()
            while t.label() == 'O':
                i += 1
                try:
                    t = result[i]
                except IndexError:
                    break
                if t.label() == 'O':
                    leaves += t.leaves()
            res.append(nltk.Tree('O', leaves))
            continue
        res.append(t)
        i += 1

    # res.draw()
    return res


def fix_dashes_slashes(sent):
    new_sent = ''
    sent_list = sent.split(' ')
    sent_list = list(filter(None, sent_list))
    l = len(sent_list)
    for i, word in enumerate(sent_list):
        if '-' in word:
            word = ' - '.join(word.split('-'))
        if '–' in word:
            word = ' – '.join(word.split('–'))
        if '/' in word:
            word = ' / '.join(word.split('/'))

        new_sent += word
        if i != l:
            new_sent += ' '

    new_sent = new_sent.replace('  ', ' ')
    return new_sent


if __name__ == '__main__':
    all_structures2()