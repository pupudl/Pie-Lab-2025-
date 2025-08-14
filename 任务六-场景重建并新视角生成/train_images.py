import os
import subprocess

# 图片集所在的绝对路径
images_path = r"D:\pycharm\gaussian-splatting\data\bear2\input"

# 上一级文件夹所在路径
folder_path = os.path.dirname(images_path)

# 脚本运行
# COLMAP估算相机位姿
command = f'python convert.py -s {folder_path}'
subprocess.run(command, shell=True)
# 模型训练脚本，模型会保存在output路径下
command = f'python train.py -s {folder_path}'
subprocess.run(command, shell=True)
