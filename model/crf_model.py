"""
Implements the CRF model class
"""
from sklearn.externals import joblib
import nltk


class Model:
    """
    Loads the CRF model and returns labels given a sentence.
    """

    def __init__(self, model_path='model.pkl'):
        """
        :param model_path: path to the model.
        """
        self.crf = joblib.load(model_path)

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

    def _word2features(self, sent, position):
        word = sent[position][0]
        postag = sent[position][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.containsDigit()': any(ch.isdigit() for ch in word),
            'word.containsPunctuation()': any(self.__is_punctuation(ch) for ch in word),
            'postag': postag,
            'postag[:2]': postag[:2],
        }

        if position > 1:
            word2 = sent[position - 2][0]
            postag2 = sent[position - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.containsDigit()': any(ch.isdigit() for ch in word),
                '-2:postag': postag2,
                '-2:postag[:2]': postag2[:2],
            })

        if position > 0:
            word1 = sent[position - 1][0]
            postag1 = sent[position - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.containsDigit()': any(ch.isdigit() for ch in word),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if position < len(sent) - 2:
            word2 = sent[position + 2][0]
            postag2 = sent[position + 2][1]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.isupper()': word2.isupper(),
                '+2:word.containsDigit()': any(ch.isdigit() for ch in word),
                '+2:postag': postag2,
                '+2:postag[:2]': postag2[:2],
            })

        if position < len(sent) - 3:
            word3 = sent[position + 3][0]
            postag3 = sent[position + 3][1]
            features.update({
                '+3:word.lower()': word3.lower(),
                '+3:word.istitle()': word3.istitle(),
                '+3:word.isupper()': word3.isupper(),
                '+3:word.containsDigit()': any(ch.isdigit() for ch in word),
                '+3:postag': postag3,
                '+3:postag[:2]': postag3[:2],
            })

        if position < len(sent) - 1:
            word1 = sent[position + 1][0]
            postag1 = sent[position + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.containsDigit()': any(ch.isdigit() for ch in word),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        if self.__is_punctuation(word):
            features['bias'] = 0

        return features

    def save(self):
        """
        Saves a .pkl file with the thrained model. This can later be loaded with the load function.
        """
        joblib.dump(self.crf, 'model.pkl')

    def load(self, model_file):
        """
        Loads a trained model into memory.
        :param model_file: path to the model
        """
        if not model_file.endswith('.pkl'):
            raise TypeError("Incorrect model file")
        else:
            self.crf = joblib.load(model_file)

    @staticmethod
    def __is_punctuation(word):
        if word in ['.', ',', '(', ')', '[', ']', '{', '}', ':', ';']:
            return True
        return False

    def sentence2features(self, sentence):
        """
        Returns a list of features for a sentence.
        :param sentence: A list of words.
        :return: A list of features for each word in the sentence.
        """
        return [self._word2features(sentence, i) for i in range(len(sentence))]

    @staticmethod
    def sentence2labels(sentence):
        """
        Labels a sentence.
        :param sentence: A sentence taken from the corpus file.
        :return: A list of labels predicted by the model.
        """
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

            if word.endswith('.') and word != '.' and i < length - 1:
                word = word[:-1]

            new_sentence += word
            if i != length:
                new_sentence += ' '

        return new_sentence
