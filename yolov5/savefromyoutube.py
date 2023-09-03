#ダウンロードしたyoutubeの動画からランダムに画像を保存する
number = 1

import cv2
import random
import os
import yt_dlp

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


if(number == 0):
    name = 'shinkansen'
    video_path = 'https://www.youtube.com/watch?v=A8n6uwJs4bA'
    dir_path = f'/home/nozaki/ML/ultralytics/pic/{name}'
    num = 500
else:
    name = 'densya'
    video_path = 'https://www.youtube.com/watch?v=h44if9TrMeI'
    dir_path = f'/home/nozaki/ML/ultralytics/pic/{name}'
    num = 500

get_random_frame(name, video_path, dir_path, num)
