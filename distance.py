import nltk, os
from sklearn.externals import joblib
from apted import helpers,apted, Config
import subprocess
import re
from model import getLabels

clf = joblib.load('model.pkl')


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

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O'])
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

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O'])
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

    tree = nltk.chunk.conllstr2tree(text, chunk_types=['DOS','UNIT','WHO', 'O'])
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


def all_structures(folder='./dosages/'):
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


if __name__ == '__main__':

    all_structures()

    first_struct = ['ablavar', 'acular-ls', 'adacel', 'addyi', 'adlyxin', 'advair-hfa', 'afluria-quadrivalent', 'alkeran-injection', 'alsuma', 'altace', 'altoprev', 'arava', 'aripiprazole-tablets', 'aristocort-forte', 'armonair-respiclick', 'aromasin', 'astepro', 'avapro', 'avonex', 'bavencio', 'belviq', 'bentyl', 'besivance', 'bextra', 'bleph', 'blocadren', 'boniva-injection', 'brisdelle', 'capoten', 'cardura', 'cerdelga', 'cisplatin', 'claforan', 'clarinex-d-24hr', 'colcigel', 'combivent', 'corlanor', 'corphedra', 'cortone', 'cozaar', 'cystaran', 'cytomel', 'darvon', 'depakote-er', 'diabinese', 'diclegis', 'diovan-hct', 'duraclon', 'durezol', 'emadine', 'ethyol', 'exondys-51', 'fareston', 'fastin', 'flomax', 'flublok', 'fluocinolone', 'fluorometholone', 'fosrenol', 'geodon', 'gilotrif', 'gralise', 'haldol', 'harvoni', 'hyqvia', 'hysingla-er', 'ibrance', 'iclusig', 'inderal-la', 'iressa', 'istodax', 'ixempra', 'jardiance', 'jenloga', 'keppra-xr', 'keveyis', 'kyprolis', 'lamisil', 'levatol', 'levo-dromoran', 'liptruzet', 'loniten', 'lovaza', 'lupron-depot-375', 'lupron-depot-75', 'meclizine-hydrochloride', 'mintezol', 'mitigare', 'movantik', 'mycelex', 'mycobutin', 'olysio', 'opdivo', 'peganone', 'pentacel', 'perforomist', 'permax', 'platinol', 'pletal', 'poly-pred', 'pondimin', 'prandimet', 'pred-forte', 'prempro', 'prestalia', 'provenge', 'provera', 'qbrelis', 'rapaflo-capsules', 'reclast', 'relistor', 'reprexain', 'requip', 'rescula', 'retrovir', 'rezulin', 'rheumatrex', 'rozerem', 'safyral', 'sanctura', 'sanctura-xr', 'sandimmune', 'saphris', 'savaysa', 'simponi-aria', 'somavert', 'sorine', 'spiriva-respimat', 'sprycel', 'stavzor', 'staxyn', 'sumavel-dosepro', 'synercid', 'synribo', 'targretin', 'tecentriq', 'timoptic', 'timoptic-in-ocudose', 'timoptic-xe', 'tobradex-st', 'torisel', 'tradjenta', 'transderm-nitro', 'travatan', 'trental', 'trulicity', 'tyvaso', 'tyzeka', 'uptravi', 'urobiotic', 'vayarin', 'vayarol', 'vectibix', 'versacloz', 'vesicare', 'vexol', 'vigamox', 'virazole', 'viread', 'vivitrol', 'vivlodex', 'xanax-xr', 'xenazine', 'xifaxan', 'xiidra', 'xtoro', 'zebeta', 'zepatier', 'zolinza']
    second_struct = ['actos', 'amaryl', 'amerge', 'baycol', 'benlysta', 'calcijex', 'celestone', 'clemastine-fumarate-tablets', 'cortef', 'cosentyx', 'definity', 'deltasone', 'depo-provera', 'dilacor-xr', 'doxorubicin-hydrochloride', 'duexis', 'duoneb', 'durlaza', 'dutrebis', 'elmiron', 'erythrocin-stearate', 'famvir', 'feraheme', 'fetzima', 'fortical', 'fuzeon', 'gynazole', 'imbruvica', 'imlygic', 'incivek', 'inderal', 'inflectra', 'invega-sustenna', 'levetiracetam', 'lopressor', 'lupaneta-pack', 'lupron-depot', 'metozolv', 'nardil', 'nascobal', 'neulasta', 'nexavar', 'nilandron', 'norvasc', 'novolin-r', 'orenitram', 'oxtellar-xr', 'pamidronate-disodium', 'pce', 'prostascint', 'rebif', 'remeron', 'remeron-soltab', 'rixubis', 'rytary', 'savella', 'sprix', 'striant', 'tenoretic', 'timoptic-in-ocudose', 'tygacil', 'vemlidy', 'viibryd', 'winrho-sdf', 'zanosar', 'zelboraf']
    third_struct = ['fioricet', 'orbivan', 'tobradex']

    for drug in second_struct:
        with open('./dosages/'+drug) as f:
            sent = f.readline()
            f.close()
        print(view_tree(sent))

    # with open('./dosages/' + 'trezix') as f:
    #     sent = f.readline()
    #     f.close()
    # view_tree(sent)