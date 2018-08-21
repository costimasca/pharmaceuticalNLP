from model.crf_model import Model
import os
import nltk


class I2B2Extractor:
    def __init__(self):
        self.model = Model('model.pkl')
        self.path = '../i2b2_data/data/'
        self.identifiers = {'DOS': 'do', 'UNIT': 'do', 'FREQ': 'f', 'PER': 'du'}
        label_files = os.listdir(self.path + 'annotations_ground_truth/pool/')
        self.label_file_dict = {name.split('.')[0]: name for name in label_files}

    def parse_file(self, file):
        with open(file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            self.label_line(line, i+1)

    def extract_all(self):
        training_set_folders = [self.path + 'training.sets.released/' + str(ind) for ind in range(1, 11)]

        for training_set in training_set_folders:
            for file in os.listdir(training_set):
                if file in self.label_file_dict.keys():
                    self.parse_file(training_set + '/' + file)

                return

    def write_output(self):
        pass

    def label_line(self, line, index):
        prediction = self.model.predict(line)
        if len(prediction) == 0:
            return
        prediction = prediction[0]

        print(prediction)
        entities = {'m': ['nm'], 'do': [], 'mo': ['nm'], 'f': [], 'du': [], 'r': ['nm'], 'ln': ['nm']}
        for i, (word, entity) in enumerate(prediction):
            if entity not in self.identifiers.keys():
                continue
            entities[self.identifiers[entity]].append((word, str(index) + ':'+ str(i)))
        print(entities.items())
        print()

    def generate_labeled_file(self, file):
        with open(self.path + 'train.test.released.8.17.09/' + file) as f:
            lines = f.readlines()

        words1 = []
        locations = []
        for i, line in enumerate(lines):
            words = line.split(' ')
            words = [w for w in words if w not in ['\n', '']]
            words = [w.replace('\n', '') if w.endswith('\n') else w for w in words]
            locations += [(i+1, n) for n in range(len(words))]
            words1 += words

        text = ' '.join([s for s in words1])
        sentences = nltk.sent_tokenize(text)

        words2 = []
        for sent in sentences:
            words = sent.split(' ')
            words2 += words

        k = 0
        new_sentences = []
        for i, sent in enumerate(sentences):
            new_sent = sent.split(' ')
            j = 0

            while j < len(new_sent):
                word = new_sent[j]

                if not word == words1[k]:
                    if j == 0 and words1[k - 1].endswith(new_sent[0]):
                        del new_sent[0]
                        j -= 1
                        k -= 1

                    if words1[k].startswith(word):

                        if len(new_sent) > j + 1:
                            if words1[k].endswith(new_sent[j+1]):
                                new_sent[j] = words1[k]
                                del new_sent[j + 1]
                                k -= 1
                        elif words1[k].endswith(sentences[i+1].split(' ')[0]):
                            new_sent[j] = words1[k]
                        elif (words1[k] + words1[k+1]).endswith(sentences[i+1].split(' ')[0]):
                            new_sent[j] = words1[k]
                k += 1
                j += 1

            tmp = []
            for word in new_sent:
                if '\t' in word:
                    word = word.replace('\t', ' ')

                tmp.append(word)
            new_sentences.append(tmp)

        words2 = []
        for sent in new_sentences:
            words2 += sent

        k = 0
        labeled_sentences = []
        for i, sent in enumerate(new_sentences):
            pos = nltk.pos_tag(sent)
            loc = locations[k:k+len(pos)]
            k += len(pos)

            labeled_sentences.append(list(zip(pos, loc)))

        labels = self.get_labels(file)

        with open('i2b2_corpus/' + file + '.tsv', 'w') as f:
            for sent in labeled_sentences:
                for tup in sent:
                    if tup[1] in labels.keys():
                        f.write(tup[0][0] + '\t' + tup[0][1] + '\t' + labels[tup[1]] +'\n')
                    else:
                        f.write(tup[0][0] + '\t' + tup[0][1] + '\t' + 'O\n')
                f.write('\n')
            f.close()

    def get_labels(self, file):
        label_file = self.label_file_dict[file]

        with open(self.path + 'annotations_ground_truth/converted.noduplicates.sorted/' + label_file) as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            labels += line.split('||')

        result = {}
        for label in labels:
            entity = label.split('=')[0]
            if len(label.split(' ')) < 3:
                continue
            else:
                if ',' in label and '...' in label:
                    positions = label.split('\" ')[1].split(',')

                    for pos in positions:
                        position = pos.split(' ')[-2:]
                        line_no = int(position[0].split(':')[0])
                        start = int(position[0].split(':')[1])
                        stop = int(position[1].split(':')[1])
                        for i in range(start, stop + 1):
                            result.update({tuple([line_no, i]): entity})

                else:
                    position = label.split(' ')[-2:]
                    line_no = int(position[0].split(':')[0])
                    start = int(position[0].split(':')[1])
                    stop = int(position[1].split(':')[1])
                    for i in range(start, stop+1):
                        result.update({tuple([line_no, i]): entity})

        return result

    def generate_corpus_files(self):
        training_set = self.path + 'train.test.released.8.17.09/'
        for file in os.listdir(training_set):
            if file in self.label_file_dict.keys():
                self.generate_labeled_file(file)

    def concatenate_corpus_files(self):
        files = os.listdir('i2b2_corpus/')
        print(files)

        lines = []
        for file in files:
            with open('i2b2_corpus/' + file) as f:
                lines.append(f.readlines())

        with open('corpus_i2b2.tsv', 'w') as f:
            for text in lines:
                f.writelines(text)
                f.write('\n')


if __name__ == '__main__':
    extractor = I2B2Extractor()
    extractor.concatenate_corpus_files()