import sys
from PyQt5 import QtWidgets, QtGui

import design
import model


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.parse)
        self.pushButton_2.clicked.connect(self.load_text)
        self.model = model.Model()
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
            'PER': '#BBCD67'
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
                             f"{self.color_dict['PER']};\">Period</span></p></body></html>")

    def load_text(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QMainWindow(), 'Open file', '~', "All files *")

        with open(fname[0]) as f:
            text = f.read(-1)
            f.close()

        self.text_doc.setHtml('<body>' + text + '</body>')

    def parse(self):
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
                rich_text += f'<nobr><label style=\"background-color:{self.color_dict[label]}\">{word}</label></nobr>'

            # to keep the full stop next to the last word
            if i < length - 2:
                rich_text += f"<label> </label>"

        rich_text += f'</body>'
        self.text_doc.setHtml(rich_text)
        return self.text_doc


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
