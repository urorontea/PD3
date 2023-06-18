import os
import shutil

# 元のフォルダのパス
folder1_path = "/home/nozaki/ML/ultralytics/pic/dataset0527"
folder2_path = "/home/nozaki/ML/ultralytics/pic/dataset0601/train"

# 新しいフォルダのパス
output_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0605"
 
# ファイル名のカウンタ
counter1 = 0  # densya_0000 のカウンタ
counter2 = 0  # shinkansen_0000 のカウンタ

# フォルダ1の画像とテキストファイルをコピー
for filename in os.listdir(folder1_path):
    if filename.endswith(".jpg"):
        if "densya" in filename:
            image_src = os.path.join(folder1_path, filename)
            image_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.jpg")
            shutil.copyfile(image_src, image_dst)

            txt_filename = filename.replace(".jpg", ".txt")
            txt_src = os.path.join(folder1_path, txt_filename)
            txt_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.txt")
            shutil.copyfile(txt_src, txt_dst)

            counter1 += 1
        elif "shinkansen" in filename:
            image_src = os.path.join(folder1_path, filename)
            image_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.jpg")
            shutil.copyfile(image_src, image_dst)

            txt_filename = filename.replace(".jpg", ".txt")
            txt_src = os.path.join(folder1_path, txt_filename)
            txt_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.txt")
            shutil.copyfile(txt_src, txt_dst)

            counter2 += 1

# フォルダ2の画像とテキストファイルをコピー
for filename in os.listdir(folder2_path):
    if filename.endswith(".jpg"):
        if "densya" in filename:
            image_src = os.path.join(folder2_path, filename)
            image_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.jpg")
            shutil.copyfile(image_src, image_dst)

            txt_filename = filename.replace(".jpg", ".txt")
            txt_src = os.path.join(folder2_path, txt_filename)
            txt_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.txt")
            shutil.copyfile(txt_src, txt_dst)

            counter1 += 1
        elif "shinkansen" in filename:
            image_src = os.path.join(folder2_path, filename)
            image_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.jpg")
            shutil.copyfile(image_src, image_dst)

            txt_filename = filename.replace(".jpg", ".txt")
            txt_src = os.path.join(folder2_path, txt_filename)
            txt_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.txt")
            shutil.copyfile(txt_src, txt_dst)

            counter2 += 1
