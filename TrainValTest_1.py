import os
import random
import shutil

# フォルダのパスと分割比率の設定
data_folder = "/home/nozaki/ML/ultralytics/pic/dataset0704/W7/cropped/"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# データセット内のファイルリストを取得
file_list = os.listdir(data_folder)
file_list.sort()

# フォルダの作成
train_folder = os.path.join(data_folder, "train")
val_folder = os.path.join(data_folder, "val")
test_folder = os.path.join(data_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 画像ファイルとテキストファイルの分割
train_densya_count = 0
val_densya_count = 0
test_densya_count = 0

train_shinkansen_count = 0
val_shinkansen_count = 0
test_shinkansen_count = 0

for img_filename in file_list:
    # 画像ファイルかどうかを確認
    if not img_filename.endswith(".jpg"):
        continue

    txt_filename = img_filename.replace(".jpg", ".txt")

    # ファイルのパスを取得
    img_path = os.path.join(data_folder, img_filename)
    txt_path = os.path.join(data_folder, txt_filename)

    # 分割先のフォルダを選択
    if "densya" in img_filename:
        if train_densya_count < train_ratio * len(file_list) / 2:
            dest_folder = train_folder
            train_densya_count += 1
        elif val_densya_count < val_ratio * len(file_list) / 2:
            dest_folder = val_folder
            val_densya_count += 1
        else:
            dest_folder = test_folder
        
    elif "shinkansen" in img_filename:
        if train_shinkansen_count < train_ratio * len(file_list) / 2:
            dest_folder = train_folder
        elif val_shinkansen_count < val_ratio * len(file_list) / 2:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        train_shinkansen_count += 1
    else:
        continue
    
    # ファイルの移動
    shutil.move(img_path, os.path.join(dest_folder, img_filename))
    shutil.move(txt_path, os.path.join(dest_folder, txt_filename))
