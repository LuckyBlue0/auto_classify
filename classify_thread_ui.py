import glob
import os
import threading
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QVBoxLayout, QWidget, QLabel, QMessageBox, QInputDialog, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import shutil
import sys

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 用于存储每个类别对应的文件路径列表
category_files_dict = {}
lock = threading.Lock()
folder_lock = threading.Lock()

class ImageClassifierThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(str)

    def __init__(self, image_files):
        super().__init__()
        self.image_files = image_files

    def process_image(self, image_file):
        try:
            image = Image.open(image_file)
            self.progress.emit(f"正在处理图片: {image_file}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class_label = model.config.id2label[predicted_class_idx]

            with lock:
                if predicted_class_label not in category_files_dict:
                    category_files_dict[predicted_class_label] = []
                category_files_dict[predicted_class_label].append(image_file)

        except Exception as e:
            self.progress.emit(f"处理图片 {image_file} 时出错: {str(e)}")

    def run(self):
        max_threads = 4
        threads = []
        
        for image_file in self.image_files:
            if len(threads) < max_threads:
                thread = threading.Thread(target=self.process_image, args=(image_file,))
                threads.append(thread)
                thread.start()
            else:
                for t in threads:
                    t.join()
                threads = []
                thread = threading.Thread(target=self.process_image, args=(image_file,))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        self.finished.emit()

class ImageClassifierUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.directory = None
        self.initUI()

    def center_window(self):
        # 获取屏幕几何信息
        screen = QApplication.primaryScreen().geometry()
        # 获取窗口几何信息
        window_geometry = self.geometry()
        
        # 计算窗口居中位置
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        
        # 移动窗口到居中位置
        self.move(x, y)

    def initUI(self):
        self.setWindowTitle("图片分类工具")
        self.setGeometry(0, 0, 600, 700)
        self.center_window()
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 15px;
                min-height: 45px;
                font-weight: 500;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLabel {
                font-size: 14px;
                color: #424242;
                padding: 8px;
            }
        """)

        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # 创建标题标签
        title_label = QLabel("AI 图片智能分类工具")
        title_label.setStyleSheet("""
            font-size: 28px;
            color: #1565C0;
            font-weight: bold;
            padding: 20px;
            background-color: white;
            border-radius: 15px;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 创建信息面板
        info_panel = QWidget()
        info_panel.setStyleSheet("""
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
        """)
        info_layout = QVBoxLayout(info_panel)

        # 创建标签
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setAlignment(Qt.AlignCenter)
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setStyleSheet("""
            font-size: 15px;
            color: #333333;
            padding: 15px;
            margin: 5px;
        """)
        info_layout.addWidget(self.folder_path_label)

        self.file_count_label = QLabel("文件夹内文件数量: 0")
        self.file_count_label.setAlignment(Qt.AlignCenter)
        self.file_count_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2196F3;
            padding: 15px;
            margin: 5px;
        """)
        info_layout.addWidget(self.file_count_label)

        layout.addWidget(info_panel)

        # 创建按钮组
        buttons_panel = QWidget()
        buttons_layout = QVBoxLayout(buttons_panel)
        buttons_layout.setSpacing(15)

        # 创建按钮
        select_folder_btn = QPushButton("选择要处理的文件夹")
        select_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        select_folder_btn.clicked.connect(self.select_folder)
        buttons_layout.addWidget(select_folder_btn)

        classify_btn = QPushButton("开始智能分类")
        classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        classify_btn.clicked.connect(self.classify_and_move_files)
        buttons_layout.addWidget(classify_btn)

        add_word_btn = QPushButton("给当前选择的文件夹下面所有.txt文件中添加文字")
        add_word_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        add_word_btn.clicked.connect(self.add_word_to_txt_files)
        buttons_layout.addWidget(add_word_btn)

        layout.addWidget(buttons_panel)

        # 状态面板
        status_panel = QWidget()
        status_panel.setStyleSheet("""
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
        """)
        status_layout = QVBoxLayout(status_panel)

        # 状态标签
        self.status_label = QLabel("等待开始...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #666666;
            font-size: 14px;
            padding: 15px;
            margin: 5px;
        """)
        status_layout.addWidget(self.status_label)

        layout.addWidget(status_panel)

    def select_folder(self):
        self.directory = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if self.directory:
            self.folder_path_label.setText(f"已选择文件夹: {self.directory}")
            file_count = len(glob.glob(os.path.join(self.directory, '*')))
            self.file_count_label.setText(f"文件夹内文件数量: {file_count}")
            self.file_count_label.setStyleSheet("color: red;")

    def classify_and_move_files(self):
        if not self.directory:
            QMessageBox.critical(self, "错误", "请先选择要处理的文件夹！")
            return

        image_files = [file for file in glob.glob(os.path.join(self.directory, '*')) 
                      if not file.endswith('.txt')]
        
        self.classifier_thread = ImageClassifierThread(image_files)
        self.classifier_thread.progress.connect(self.update_status)
        self.classifier_thread.finished.connect(self.finish_classification)
        self.classifier_thread.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def finish_classification(self):
        try:
            # 清空旧的分类文件夹
            for category in category_files_dict.keys():
                category_folder = os.path.join(self.directory, category)
                with folder_lock:
                    os.makedirs(category_folder, exist_ok=True)

            # 使用集合来跟踪已处理的文件
            processed_files = set()

            for category, file_paths in category_files_dict.items():
                category_folder = os.path.join(self.directory, category)
                
                for file_path in file_paths:
                    try:
                        # 检查文件是否存在且未被处理
                        if os.path.exists(file_path) and file_path not in processed_files:
                            file_name = os.path.basename(file_path)
                            destination_path = os.path.join(category_folder, file_name)
                            
                            # 如果目标文件已存在，添加数字后缀
                            counter = 1
                            while os.path.exists(destination_path):
                                name, ext = os.path.splitext(file_name)
                                destination_path = os.path.join(category_folder, f"{name}_{counter}{ext}")
                                counter += 1

                            # 移动文件
                            shutil.move(file_path, destination_path)
                            processed_files.add(file_path)

                            # 处理对应的txt文件
                            txt_file_path = os.path.splitext(file_path)[0] + '.txt'
                            if os.path.exists(txt_file_path):
                                txt_name = os.path.basename(txt_file_path)
                                txt_destination_path = os.path.join(category_folder, txt_name)
                                
                                # 如果目标txt文件已存在，添加数字后缀
                                counter = 1
                                while os.path.exists(txt_destination_path):
                                    name, ext = os.path.splitext(txt_name)
                                    txt_destination_path = os.path.join(category_folder, f"{name}_{counter}{ext}")
                                    counter += 1

                                shutil.move(txt_file_path, txt_destination_path)

                    except Exception as e:
                        self.status_label.setText(f"移动文件时出错: {str(e)}")
                        continue

            QMessageBox.information(self, "完成", "图片分类及文件移动操作已完成！")
            
            # 清空分类字典
            category_files_dict.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出现错误: {str(e)}")

    def add_word_to_txt_files(self):
        if not self.directory:
            QMessageBox.critical(self, "错误", "请先选择要处理的文件夹！")
            return

        word, ok = QInputDialog.getText(self, "输入单词", "请输入要添加的单词:")
        if ok and word:
            for root, dirs, files in os.walk(self.directory):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r+') as f:
                            content = f.read()
                            f.seek(0, 0)
                            f.write(word + content)

            QMessageBox.information(self, "完成", "已给所有.txt文件添加单词！")

def main():
    app = QApplication(sys.argv)
    ui = ImageClassifierUI()
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()