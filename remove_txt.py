import os
base_name = "/home/nozaki/ML/ultralytics/pic/dataset"
folder_path =base_name + "0704/W7/cropped"  # フォルダのパスを指定

# フォルダ内のファイルを取得
files = os.listdir(folder_path)


for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".txt"):
        os.remove(file_path)
        print(f"{file_name}を削除しました")
