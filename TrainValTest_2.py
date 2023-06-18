import os
import random
import shutil

# フォルダのパスと分割比率の設定
data_folder = "/home/nozaki/ML/ultralytics/pic/sozai/img/cropped"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# データセット内のファイルリストを取得
file_list = os.listdir(data_folder)
file_list.sort()
#print(file_list,'\n')

# フォルダの作成
train_folder = os.path.join(data_folder, "train")
val_folder = os.path.join(data_folder, "val")
test_folder = os.path.join(data_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# データセットの分割
densyaIMG_files = [file for file in file_list if "densya" in file and "jpg" in file]
densyaIMG_files.sort()
densyaTXT_files = [file for file in file_list if "densya" in file and "txt" in file]
densyaTXT_files.sort()
shinkansenIMG_files = [file for file in file_list if "shinkansen" in file and "jpg" in file]
shinkansenIMG_files.sort()
shinkansenTXT_files = [file for file in file_list if "shinkansen" in file and "txt" in file]
shinkansenTXT_files.sort()

#random.shuffle(densya_files)
#random.shuffle(shinkansen_files)

num_densya = len(densyaIMG_files)
num_shinkansen = len(shinkansenIMG_files)

num_train_densya = int(num_densya * train_ratio)
num_val_densya = int(num_densya * val_ratio)
num_test_densya = num_densya - num_train_densya - num_val_densya

num_train_shinkansen = int(num_shinkansen * train_ratio)
num_val_shinkansen = int(num_shinkansen * val_ratio)
num_test_shinkansen = num_shinkansen - num_train_shinkansen - num_val_shinkansen

# 電車データの分割
for i in range(num_densya):
    img_filename = densyaIMG_files[i]
    #print("img_filenameは、",img_filename)
    txt_filename = img_filename.replace(".jpg", ".txt")
    #print("txt_filenameは、",txt_filename)
    
    # ファイルのパスを取得
    img_path = os.path.join(data_folder, img_filename)
    txt_path = os.path.join(data_folder, txt_filename)
    
    # 分割先のフォルダを選択
    if i < num_train_densya:
        #dest_folder = os.path.join(train_folder, "densya")
        dest_folder = train_folder
    elif i < num_train_densya + num_val_densya:
        #dest_folder = os.path.join(val_folder, "densya")
        dest_folder = val_folder
    else:
        #dest_folder = os.path.join(test_folder, "densya")
        dest_folder = test_folder
    
    # ファイルの移動
    shutil.move(img_path, os.path.join(dest_folder, img_filename))
    #print(img_filename,"を移動しました")
    shutil.move(txt_path, os.path.join(dest_folder, txt_filename))

# 新幹線データの分割
for i in range(num_shinkansen):
    img_filename = shinkansenIMG_files[i]
    txt_filename = img_filename.replace(".jpg", ".txt")
    
    # ファイルのパスを取得
    img_path = os.path.join(data_folder, img_filename)
    txt_path = os.path.join(data_folder, txt_filename)
    
    # 分割先のフォルダを選択
    if i < num_train_shinkansen:
        #dest_folder = os.path.join(train_folder, "shinkansen")
        dest_folder = train_folder
    elif i < num_train_shinkansen + num_val_shinkansen:
        #dest_folder = os.path.join(val_folder, "shinkansen")
        dest_folder = val_folder
    else:
        #dest_folder = os.path.join(test_folder, "shinkansen")
        dest_folder = test_folder

    # ファイルの移動
    shutil.move(img_path, os.path.join(dest_folder, img_filename))
    shutil.move(txt_path, os.path.join(dest_folder, txt_filename))
    
