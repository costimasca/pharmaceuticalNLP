import crfTrainer
import random
from crfUtil import loadCorpus


def performance_measure(corpus_file="corp.tsv", ten_fold=False, verbose=False):
    """
    Measures the performance on a given corpus file.
    ten_fold parameter changes the behavior of the function, performing a
    10-fold cross validation if True, otherwise only measuring the performance
    once, on a 9 to 1 ratio train-test dataset.

    Parameters
    ----------
    corpus_file : optional
        the name of the corpus file. defaults to "corp.tsv"
    ten_fold :
        if True, performs 10-fold cross validation
    verbose:
        if True, prints the performance for each fold
    Returns
    -------
    string
        precision, recall and f1-score for all labels

    """
    sent = loadCorpus(corpus_file)
    random.shuffle(sent)
    l = int(len(sent) / 10)

    sets = []

    for i in range(9):
        sets.append(random.sample(sent, l))
        sent = [s for s in sent if s not in sets[i]]

    sets.append(sent)

    per_precision = 0
    who_precision = 0
    unit_precision = 0
    dos_precision = 0
    freq_precision = 0

    per_recall = 0
    who_recall = 0
    unit_recall = 0
    dos_recall = 0
    freq_recall = 0

    per_f1 = 0
    who_f1 = 0
    unit_f1 = 0
    dos_f1 = 0
    freq_f1 = 0

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for s in sets:
        test = s
        train = [sentence for set in sets for sentence in set if set != s]
        data = crfTrainer.gen_model(train, test, False)
        if verbose:
            print(data)

        data = data.split('     ')

        per_precision += float(data[4])
        who_precision += float(data[9])
        unit_precision += float(data[14])
        dos_precision += float(data[19])
        freq_precision += float(data[24])

        per_recall += float(data[5])
        who_recall += float(data[10])
        unit_recall += float(data[15])
        dos_recall += float(data[20])
        freq_recall += float(data[25])

        per_f1 += float(data[6])
        who_f1 += float(data[11])
        unit_f1 += float(data[16])
        dos_f1 += float(data[21])
        freq_f1 += float(data[26])

        total_precision += float(data[28])
        total_recall += float(data[29])
        total_f1 += float(data[30])

        if not ten_fold:
            break

    if ten_fold:
        per_precision /= 10
        who_precision /= 10
        unit_precision /= 10
        dos_precision /= 10
        freq_precision /= 10

        per_recall /= 10
        who_recall /= 10
        unit_recall /= 10
        dos_recall /= 10
        freq_recall /= 10

        per_f1 /= 10
        who_f1 /= 10
        unit_f1 /= 10
        dos_f1 /= 10
        freq_f1 /= 10

        total_precision /= 10
        total_recall /= 10
        total_f1 /= 10

    res = '        precision    recall    f1-score\n'
    res += (
        'PER     ' + "{:.3f}".format(per_precision) + '\t\t' + "{:.3f}".format(per_recall) + '\t\t' + "{:.3f}".format(
            per_f1) + '\n')
    res += (
        'WHO    ' + "{:.3f}".format(who_precision) + '\t\t' + "{:.3f}".format(who_recall) + '\t\t' + "{:.3f}".format(
            who_f1) + '\n')
    res += (
        'UNIT    ' + "{:.3f}".format(unit_precision) + '\t\t' + "{:.3f}".format(unit_recall) + '\t\t' + "{:.3f}".format(
            unit_f1) + '\n')
    res += (
        'DOS     ' + "{:.3f}".format(dos_precision) + '\t\t' + "{:.3f}".format(dos_recall) + '\t\t' + "{:.3f}".format(
            dos_f1) + '\n')
    res += (
        'FREQ     ' + "{:.3f}".format(freq_precision) + '\t\t' + "{:.3f}".format(freq_recall) + '\t\t' + "{:.3f}".format(
            freq_f1) + '\n')

    res += (
        'AVG     ' + "{:.3f}".format(total_precision) + '\t\t' + "{:.3f}".format(
            total_recall) + '\t\t' + "{:.3f}".format(
            total_f1) + '\n')

    return res


def error_distribution(corpus_file="corp.tsv"):
    sent = loadCorpus(corpus_file)
    random.shuffle(sent)
    l = int(len(sent) / 10)

    sets = []

    for i in range(9):
        sets.append(random.sample(sent, l))
        sent = [s for s in sent if s not in sets[i]]

    sets.append(sent)

    unit_precision = []
    dos_precision = []
    unit_recall = []
    dos_recall = []
    unit_f1 = []
    dos_f1 = []
    total_precision = []
    total_recall = []
    total_f1 = []

    for s in sets:
        test = s
        train = [sentence for set in sets for sentence in set if set != s]
        data = crfTrainer.gen_model(train, test, False)

        print(data)

        data = data.split('     ')

        unit_precision.append(float(data[4]))
        dos_precision.append(float(data[9]))
        unit_recall.append(float(data[5]))
        dos_recall.append(float(data[10]))
        unit_f1.append(float(data[6]))
        dos_f1.append(float(data[10]))
        total_precision.append(float(data[13]))
        total_recall.append(float(data[14]))
        total_f1.append(float(data[15]))

    print(unit_precision)
    print(dos_precision)
    print(unit_recall)
    print(dos_recall)
    print(unit_f1)
    print(dos_f1)
    print(total_precision)
    print(total_recall)
    print(total_f1)

if __name__ == '__main__':
    print(performance_measure(ten_fold=True, verbose=True))
