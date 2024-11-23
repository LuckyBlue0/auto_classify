from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from PIL import Image
import sys
import os

class AvifConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('AVIF to PNG Converter')
        self.setGeometry(300, 300, 400, 200)
        
        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建标签
        self.label = QLabel('选择AVIF文件进行转换', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # 创建按钮
        self.select_button = QPushButton('选择文件', self)
        self.select_button.clicked.connect(self.select_files)
        layout.addWidget(self.select_button)
        
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择AVIF文件",
            "",
            "AVIF files (*.avif)"
        )
        
        if files:
            for file_path in files:
                try:
                    # 转换文件
                    img = Image.open(file_path)
                    output_path = os.path.splitext(file_path)[0] + '.png'
                    img.save(output_path, 'PNG')
                    self.label.setText(f'成功转换: {os.path.basename(file_path)}')
                except Exception as e:
                    self.label.setText(f'转换失败: {str(e)}')

def main():
    app = QApplication(sys.argv)
    converter = AvifConverter()
    converter.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()