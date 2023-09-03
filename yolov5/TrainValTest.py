import os
import random
import shutil

# フォルダのパスと分割比率の設定
data_folder = "/home/nozaki/ML/ultralytics/pic/dataset0605"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

#各フォルダのパス
train_folder  = "/home/nozaki/ML/ultralytics/pic/dataset0605/train"
val_folder = "/home/nozaki/ML/ultralytics/pic/dataset0605/val"
test_folder = "/home/nozaki/ML/ultralytics/pic/dataset0605/test"

# データセット内のファイルリストを取得
file_list = os.listdir(data_folder)
file_list.sort()

# データセットのサイズと分割数を計算
num_files = len(file_list) // 2
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val

# データセットの分割
for i in range(num_files):
    img_filename = file_list[i * 2]
    txt_filename = file_list[i * 2 + 1]

    # ファイルのパスを取得
    img_path = os.path.join(data_folder, img_filename)
    txt_path = os.path.join(data_folder, txt_filename)

    # 分割先のフォルダを選択
    if i < num_train:
        dest_folder = train_folder
    elif i < num_train + num_val:
        dest_folder = val_folder
    else:
        dest_folder = test_folder

    # ファイルの移動
    shutil.move(img_path, os.path.join(dest_folder, img_filename))
    shutil.move(txt_path, os.path.join(dest_folder, txt_filename))
