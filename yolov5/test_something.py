#ダウンロードしたyoutubeの動画からランダムに画像を保存する

import cv2
import random
import os
import yt_dlp
import glob
import torch

def get_random_frame(name, video_path, dir_path, img_num):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': '%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_path, download=True)
        filename = ydl.prepare_filename(info_dict)

    video_path = filename  # ダウンロードされたファイルのパスを使用

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)

    for i in range(img_num):
        #name_1 = name + '_' + str(i) + '.jpg'
        name_1 = f'{name}_{str(i).zfill(4)}.jpg'
        #filename = f'train_{str(i).zfill(4)}.jpg'

        file_path = os.path.join(dir_path, name_1)
        os.makedirs(dir_path, exist_ok=True)
        random_frame_number = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(file_path, frame)
            print(f"Saved random frame {random_frame_number} as {name_1}")
        else:
            print(f"Failed to read frame {random_frame_number}")


input_folder_path =  "/home/nozaki/ML/ultralytics/pic/sozai/img/"
output_folder_path = "/home/nozaki/ML/ultralytics/pic/sozai/img/densya"

# YOLOv5sモデルをロードする
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 入力画像フォルダ内のすべての画像を処理する
for i, img_path in enumerate(glob.glob(os.path.join(input_folder_path, '*.jpg'))):
    # 画像を読み込む
    img = cv2.imread(img_path)

    # YOLOv5sモデルで物体検出を実行する
    result = model(img)

    # 物体検出結果をPandasのDataFrame形式で取得する
    obj = result.pandas().xyxy[0]

    # 検出された電車のバウンディングボックスの中身を別のフォルダに保存する
    for j in range(len(obj)):
        name = obj.name[j]
        xmin = obj.xmin[j]
        ymin = obj.ymin[j]
        xmax = obj.xmax[j]
        ymax = obj.ymax[j]

        # 電車として認識されたモノだけ保存する
        if name == 'train':
            # バウンディングボックスの情報を元に、元画像から切り抜く領域を決定する
            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

            # 保存先のパスを生成する
            #filename = f'train_{str(i).zfill(4)}_{str(j).zfill(4)}.jpg'
            filename = f'{type}_{str(i).zfill(4)}.jpg'
            save_path = os.path.join(output_folder_path, filename)

            # 切り抜いた画像を保存する
            cv2.imwrite(save_path, cropped_img)

            print(f'Saved {filename}')


#電車の画像が欲しい時
name = "densya"
video_path = "https://www.youtube.com/watch?v=h44if9TrMeI"
dir_path = f"/home/nozaki/ML/ultralytics/pic/sozai/img/{name}"
num = 10
get_random_frame(name, video_path, dir_path, num)

