import sys
from PyQt5 import QtWidgets

import design
import model


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.parse)
        self.model = model.Model()
        self.color_dict = {
            'O': '#FFFFFF',
            'DOS': '#EE964B',
            'UNIT': '#F95738',
            'WHO': '#FAF0CA',
            'FREQ': '#F4D35E',
            'PER': '#0D3B66'
        }

    def parse(self):
        text = self.textEdit.toPlainText()

        parsed_sentences = self.model.label(text)
        self.textEdit.clear()
        for sentence in parsed_sentences:
            print(sentence)
            self.textEdit.append(self.__get_rich_text(sentence))

    def __get_rich_text(self, parsed_text):
        rich_text = ''

        for i, (word, label) in enumerate(parsed_text):
            rich_text += f'<span style=\"background-color: {self.color_dict[label]}\">{word} </span>'

        return rich_text


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
