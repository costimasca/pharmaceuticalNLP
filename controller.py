import sys
from PyQt5 import QtWidgets, QtGui

import design
import model


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.parse)
        self.model = model.Model()
        self.css = '''
        label {
            font-style: normal;
            padding-right: 4 px;
        }
        body {
            background-color: gray;
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
            'WHO': '#FAF0CA',
            'FREQ': '#F4D35E',
            'PER': '#1D7AD1'
        }

    def parse(self):
        text = self.textEdit.toPlainText()

        parsed_sentences = self.model.predict(text)
        self.textEdit.clear()
        for sentence in parsed_sentences:
            self.textEdit.setDocument(self.__get_rich_text(sentence))

    def __get_rich_text(self, parsed_text):
        rich_text = f'<body><nobr>'
        length = len(parsed_text)
        for i, (word, label) in enumerate(parsed_text):
            if label == 'O':
                rich_text += f'<span>{word}</span>'
            else:
                rich_text += f'<span style=\"background-color:{self.color_dict[label]}\">{word}</span>'

            # to keep the full stop next to the last word
            if i < length - 2:
                rich_text += f"<span> </span>"

        rich_text += f'</nobr></body>'
        self.text_doc.setHtml(rich_text)
        return self.text_doc


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
