import os


densya_type = "321"
folder_path = f"/home/nozaki/ultralytics/datasets/densya_type/{densya_type}/"


file_num = 0

file_list = os.listdir(folder_path)
for filename in file_list:
    if filename.endswith((".png",".jpg",".jfif")):
        new_filename = f"{densya_type}_{file_num}.png"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        file_num += 1

        print(f"{old_filepath} --- {new_filepath}\n")

print(f"file_num = {file_num}")
