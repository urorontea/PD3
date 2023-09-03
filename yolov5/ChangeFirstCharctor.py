import os

folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0602/val"

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            if line.startswith("0"):
                modified_lines.append("1" + line[1:])
            elif line.startswith("1"):
                modified_lines.append("0" + line[1:])
            else:
                modified_lines.append(line)

        with open(file_path, "w") as file:
            file.writelines(modified_lines)
