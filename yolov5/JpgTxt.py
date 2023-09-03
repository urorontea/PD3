import os

folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/415/cropped/"  # フォルダのパスを指定

# フォルダ内のファイルを取得
files = os.listdir(folder_path)

# ファイルの拡張子ごとにグループ分け
image_files = []
text_files = []
classes_txt = []
for file in files:
    if file.endswith(".jpg"):
        image_files.append(file)
    elif file == os.path.basename("classes"):
        classes_txt.append(file)
    elif file.endswith(".txt"):
        text_files.append(file)

# 片方しか存在しないファイルを削除
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    corresponding_text_file = base_name + ".txt"
    if corresponding_text_file not in text_files:
        file_path = os.path.join(folder_path, image_file)
        os.remove(file_path)
        print(f"ファイル {image_file} を削除しました")

for text_file in text_files:
    base_name = os.path.splitext(text_file)[0]
    if base_name == "classes":
        continue
    corresponding_image_file = base_name + ".jpg"  # または ".png" や ".jpeg"
    if corresponding_image_file not in image_files:
        file_path = os.path.join(folder_path, text_file)
        os.remove(file_path)
        print(f"ファイル {text_file} を削除しました")
