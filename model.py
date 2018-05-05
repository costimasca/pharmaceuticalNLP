from sklearn.externals import joblib
import nltk


class Model:
    """
    Loads the CRF model and returns labels given a sentence.
    """

    def __init__(self):
        self.crf = joblib.load('model.pkl')

    def predict(self, text):
        """
        Labels the named entities in the given text.
        :param text: A string representation of the sentence or sentences.
        :return: A list of sentences, each containing (word, label) tuples
        """

        new_sentences = []
        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            clean_sentence = self.__pre_process__(sentence)
            word_list = nltk.word_tokenize(clean_sentence)
            tagged_words = nltk.pos_tag(word_list)
            sentence = self.sentence2features(tagged_words)

            labels = self.crf.predict([sentence])[0]

            new_sentences.append(list(zip(word_list, labels)))

        return new_sentences

    def __word2features__(self, sent, i):
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
            'word.ispunctuation()': self.__is_punctuation(word),
            'postag': postag,
            'postag[:2]': postag[:2],
        }

        if i > 1:
            word2 = sent[i - 2][0]
            postag2 = sent[i - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.isdigit()': word2.isdigit() or postag2 == 'CD',
                '-2:postag': postag2,
                '-2:postag[:2]': postag2[:2],
            })

        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
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

        if i < len(sent) - 2:
            word2 = sent[i + 2][0]
            postag2 = sent[i + 2][1]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.isupper()': word2.isupper(),
                '+2:word.isdigit()': word2.isdigit() or postag2 == 'CD',
                '+2:postag': postag2,
                '+2:postag[:2]': postag2[:2],
            })

        if i < len(sent) - 3:
            word3 = sent[i + 3][0]
            postag3 = sent[i + 3][1]
            features.update({
                '+3:word.lower()': word3.lower(),
                '+3:word.istitle()': word3.istitle(),
                '+3:word.isupper()': word3.isupper(),
                '+3:word.isdigit()': word3.isdigit() or postag3 == 'CD',
                '+3:postag': postag3,
                '+3:postag[:2]': postag3[:2],
            })

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
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

        if self.__is_punctuation(word):
            features['bias'] = 0

        return features

    def save(self):
        joblib.dump(self.crf, 'model.pkl')

    @staticmethod
    def __is_punctuation(word):
        if word in ['.', ',', '(', ')', '[', ']', '{', '}', ':', ';']:
            return True
        return False

    def sentence2features(self, sentence):
        return [self.__word2features__(sentence, i) for i in range(len(sentence))]

    @staticmethod
    def sentence2labels(sentence):
        return [label for token, pos_tag, label in sentence]

    @staticmethod
    def __sentence2tokens__(sentence):
        return [token for token, pos_tag, label in sentence]

    @staticmethod
    def __pre_process__(sentence):
        sentence_list = nltk.word_tokenize(sentence)
        length = len(sentence_list)
        new_sentence = ''
        for i, word in enumerate(sentence_list):
            if '-' in word:
                word = ' - '.join(word.split('-'))

            if '–' in word:
                word = ' – '.join(word.split('–'))

            if '/' in word:
                word = ' / '.join(word.split('/'))

            if word.endswith('.') and not word == '.' and i < length - 1:
                word = word[:-1]

            new_sentence += word
            if i != length:
                new_sentence += ' '

        return new_sentence
