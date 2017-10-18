import nltk, os


def explore(grammar, corpus):
    l = []
    files = []

    fileList = sorted(os.listdir(corpus))
    cp = nltk.RegexpParser(grammar)

    for file in fileList:
        with open(corpus + file, 'r') as f:
            sent = f.read()
            f.close()
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        tree = cp.parse(sent)

        for subtree in tree.subtrees():
            if subtree.label() == 'DOSAGE':
                # if(subtree.leaves()[0][0] == 'is'): #for group1
                if (subtree.leaves()[0][0] == 'is'):  # for group2
                    l.append(subtree)
                    files.append(file)

    return l, files


def parseString(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)

    return sent


def convertForNGram(grammar, path):
    l = os.listdir(path)
    l = sorted(l)
    cp = nltk.RegexpParser(grammar)

    for file in l:
        with open(path + file, 'r') as f:
            sent = f.read()
            f.close()
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        tree = cp.parse(sent)

        for subtree in tree.subtrees():
            print(subtree)
        input()
