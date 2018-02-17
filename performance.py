import crfTry, random
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

    unit_precision = 0
    dos_precision = 0
    unit_recall = 0
    dos_recall = 0
    unit_f1 = 0
    dos_f1 = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for s in sets:
        test = s
        train = [sentence for set in sets for sentence in set if set != s]
        data = crfTry.gen_model(train, test, False)
        if verbose:
            print(data)

        data = data.split('     ')

        unit_precision += float(data[4])
        dos_precision += float(data[9])
        unit_recall += float(data[5])
        dos_recall += float(data[10])
        unit_f1 += float(data[6])
        dos_f1 += float(data[10])
        total_precision += float(data[13])
        total_recall += float(data[14])
        total_f1 += float(data[15])

        if not ten_fold:
            break

    if ten_fold:
        unit_precision /= 10
        dos_precision /= 10
        unit_recall /= 10
        dos_recall /= 10
        unit_f1 /= 10
        dos_f1 /= 10
        total_precision /= 10
        total_recall /= 10
        total_f1 /= 10

    res = '        precision    recall    f1-score\n'
    res += (
        'UNIT    ' + "{:.3f}".format(unit_precision) + '\t\t' + "{:.3f}".format(unit_recall) + '\t\t' + "{:.3f}".format(
            unit_f1) + '\n')
    res += (
        'DOS     ' + "{:.3f}".format(dos_precision) + '\t\t' + "{:.3f}".format(dos_recall) + '\t\t' + "{:.3f}".format(
            dos_f1) + '\n')
    res += (
        'AVG     ' + "{:.3f}".format(total_precision) + '\t\t' + "{:.3f}".format(
            total_recall) + '\t\t' + "{:.3f}".format(
            total_f1) + '\n')

    return res


if __name__ == '__main__':
    print(performance_measure(ten_fold=True,verbose=True))
