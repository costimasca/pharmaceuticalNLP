#!/usr/bin/env python3

from sklearn.externals import joblib
import nltk


class Model:
    """
    Loads the CRF model and returns labels given a sentence.
    """

    def __init__(self):
        self.clf = joblib.load('model.pkl')

    def label(self, text):
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
            sentence = self.__sentence2features__(tagged_words)

            labels = self.clf.predict([sentence])[0]

            new_sentences.append(list(zip(word_list, labels)))

        return new_sentences

    @staticmethod
    def __word2features__(sentence, i):
        word = sentence[i][0]
        pos_tag = sentence[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'pos_tag': pos_tag,
            'pos_tag[:2]': pos_tag[:2],
        }

        if i > 1:
            word2 = sentence[i - 2][0]
            pos_tag2 = sentence[i - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:pos_tag': pos_tag2,
                '-2:pos_tag[:2]': pos_tag2[:2],
            })

        if i > 0:
            word1 = sentence[i - 1][0]
            pos_tag1 = sentence[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:pos_tag': pos_tag1,
                '-1:pos_tag[:2]': pos_tag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 2:
            word2 = sentence[i + 2][0]
            pos_tag2 = sentence[i + 2][1]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.isupper()': word2.isupper(),
                '+2:pos_tag': pos_tag2,
                '+2:pos_tag[:2]': pos_tag2[:2],
            })

        if i < len(sentence) - 3:
            word3 = sentence[i + 3][0]
            pos_tag3 = sentence[i + 3][1]
            features.update({
                '+3:word.lower()': word3.lower(),
                '+3:word.istitle()': word3.istitle(),
                '+3:word.isupper()': word3.isupper(),
                '+3:pos_tag': pos_tag3,
                '+3:pos_tag[:2]': pos_tag3[:2],
            })

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][0]
            pos_tag1 = sentence[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:pos_tag': pos_tag1,
                '+1:pos_tag[:2]': pos_tag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    def __sentence2features__(self, sentence):
        return [self.__word2features__(sentence, i) for i in range(len(sentence))]

    @staticmethod
    def __sentence2labels__(sentence):
        return [label for token, pos_tag, label in sentence]

    @staticmethod
    def __sentence2tokens__(sentence):
        return [token for token, pos_tag, label in sentence]

    @staticmethod
    def __pre_process__(sentence):
        sentence_list = nltk.word_tokenize(sentence)
        l = len(sentence_list)
        new_sentence = ''
        for i, word in enumerate(sentence_list):
            if '-' in word:
                word = ' - '.join(word.split('-'))

            if '–' in word:
                word = ' – '.join(word.split('–'))

            if '/' in word:
                word = ' / '.join(word.split('/'))

            if word.endswith('.') and i < l - 1:
                word = word[:-1]

            new_sentence += word
            if i != l:
                new_sentence += ' '

        return new_sentence
