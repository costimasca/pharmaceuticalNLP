"""
Controller module for the application's interface.
"""
import sys
sys.path.append('../')
import threading
import subprocess

from PyQt5 import QtWidgets, QtGui

from view import design
from model.crf_model import Model
from model.crf_trainer import Trainer


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    """
    MVC controller.
    """
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.setupUi(self)

        self.model = Model()
        self.trainer = Trainer()

        self.pushButton.clicked.connect(self.__extract__)
        self.pushButton_2.clicked.connect(self.__load_text__)
        self.pushButton_3.clicked.connect(self.__train_model__)
        self.pushButton_4.clicked.connect(self.__load_model__)

        self.css = '''
        label {
            font-style: normal;
            padding-right: 4 px;
        }
        '''
        self.text_doc = QtGui.QTextDocument()
        self.text_doc.setDefaultStyleSheet(self.css)
        self.text_doc.setHtml('<body></body>')
        self.textEdit.setDocument(self.text_doc)
        self.color_dict = {
            'O': '#FFFFFF',
            'DOS': '#EE964B',
            'UNIT': '#F95738',
            'WHO': '#8F78AD',
            'FREQ': '#D4BA6A',
            'DUR': '#BBCD67'
        }

        self.label.setText(f"<html><head/><body><p><span style=\" background-color:"
                           f"{self.color_dict['DOS']};\">Dosage</span></p></body></html>")
        self.label_2.setText(f"<html><head/><body><p><span style=\" background-color:"
                             f"{self.color_dict['UNIT']};\">Unit</span></p></body></html>")
        self.label_3.setText(f"<html><head/><body><p><span style=\" background-color:"
                             f"{self.color_dict['WHO']};\">Who</span></p></body></html>")
        self.label_4.setText(f"<html><head/><body><p><span style=\" background-color:"
                             f"{self.color_dict['FREQ']};\">Frequency</span></p></body></html>")
        self.label_5.setText(f"<html><head/><body><p><span style=\" background-color:"
                             f"{self.color_dict['DUR']};\">Period</span></p></body></html>")

    def __load_text__(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(
            QtWidgets.QFileDialog(), 'Open file', '~', "All files *")[0]
        if not file_path == '':
            with open(file_path) as file:
                text = file.read(-1)
                file.close()

            self.text_doc.setHtml('<body>' + text + '</body>')

    def __train_finished__(self, value):
        print('received result: ' + str(value))
        self.pushButton_3.setText("Train")
        self.pushButton_3.setEnabled(True)

    def __train_model__(self):
        def __train_in_thread__(data_set):
            process = subprocess.Popen(['python', 'model/crf_trainer.py', data_set]
                                       , stdout=subprocess.PIPE)
            out, _ = process.communicate()
            process.wait()
            self.__train_finished__(out)

        data_set = QtWidgets.QFileDialog.getOpenFileName(
            QtWidgets.QFileDialog(), 'Select data set', '~', "*.tsv")[0]

        thread = threading.Thread(target=__train_in_thread__, args=(data_set,))
        thread.start()

        self.pushButton_3.setText("Training...")
        self.pushButton_3.setEnabled(False)

    def __load_model__(self):
        model_file = QtWidgets.QFileDialog.getOpenFileName(
            QtWidgets.QFileDialog(), 'Select model', '~', "*.pkl")[0]
        if model_file:
            self.model.load(model_file)

    def __extract__(self):
        text = self.textEdit.toPlainText()

        parsed_sentences = self.model.predict(text)
        self.textEdit.clear()

        labeled_text = []
        for sentence in parsed_sentences:
            labeled_text += sentence

        self.textEdit.setDocument(self.__get_rich_text(labeled_text))

    def __get_rich_text(self, parsed_text):
        rich_text = f'<body>'
        length = len(parsed_text)
        for i, (word, label) in enumerate(parsed_text):
            if label == 'O':
                rich_text += f'<label>{word}</label>'
            else:
                label = label.split('-')[1]
                rich_text += f'<nobr><label style=\"background-color:' \
                             f'{self.color_dict[label]}\">{word}</label></nobr>'

            # to keep the full stop next to the last word
            if i < length - 2:
                rich_text += f"<label> </label>"

        rich_text += f'</body>'
        self.text_doc.setHtml(rich_text)
        return self.text_doc


if __name__ == '__main__':
    APP = QtWidgets.QApplication(sys.argv)

    FORM = App()
    FORM.show()

    APP.exec_()

