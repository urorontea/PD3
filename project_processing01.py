from ultralytics import YOLO
import cv2
import random
import os
import yt_dlp
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
#from google.colab.patches import cv2_imshow
from pytube import YouTube

class project_processing:

    #動画のタイトルを取得する
    def get_video_title(self, url):
        yt = YouTube(url)
        title = yt.title
        return title
    
    def get_random_frame(self, name, youtube_url, dir_path, img_num):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': '%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
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

            #出力フォルダが存在しない場合は作成する
            os.makedirs(dir_path, exist_ok=True)
            random_frame_number = random.randint(0, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(file_path, frame)
                print(f"Saved random frame {random_frame_number} as {name_1}")
            else:
                print(f"Failed to read frame {random_frame_number}")
        print(f"saved at {dir_path}")


    def croping(self, train_type, number, input_folder_path, output_folder_path):
        
        #モデルをロードする
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        
        # 画像として保存するよーん
        # img:加工させる画像、output_folder_path:出力するフォルダのパス
        # name:識別するものの名前、
        # 入力されたモデルで物体検出を実行する
        # 拡張子が.jpgの画像ファイルを取得
        image_files = glob.glob(os.path.join(input_folder_path, '*.jpg')) 

        # 出力フォルダが存在しない場合は作成する
        os.makedirs(output_folder_path, exist_ok=True)

        #通し番号
        count = 0

        for image_file in image_files:
            # 画像の読み込み
            image = cv2.imread(image_file)

            # 物体検出の実行
            result = model(image)

            # 物体検出結果をPandasのDataFrame形式で取得する
            obj = result.pandas().xyxy[0]

            # 検出された電車のバウンディングボックスの中身を別のフォルダに保存する
            for j in range(len(obj)):

                # バウンディングボックスの情報を取得
                name = obj.name[j]
                xmin = obj.xmin[j]
                ymin = obj.ymin[j]
                xmax = obj.xmax[j]
                ymax = obj.ymax[j]

                score = obj.confidence[j]

                
                if name == "train" and score >= 0.6:

                    # 切り抜いた領域を取得
                    cropped_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]

                    #バウンディングボックスを描画する
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    
                    # 保存先のパスを生成
                    img_filename = f"{train_type}_{str(count).zfill(4)}.jpg"
                    save_path = os.path.join(output_folder_path, img_filename)

                    # 切り抜いた画像を保存
                    cv2.imwrite(save_path, cropped_img)
                    print(f"Saved {img_filename}")
                    
                    # 画像のサイズとオブジェクトの情報をテキストファイルに保存
                    #アノテーションをしているよーん
                    
                    info = f"{number} 0.0 0.0 1.0 1.0"

                    text_filename = f"{train_type}_{str(count).zfill(4)}.txt"
                    text_save_path = os.path.join(output_folder_path, text_filename)
                    with open(text_save_path, "w") as f:
                        #f.write(f"{cropped_img.shape[1]} {cropped_img.shape[0]}\n")  # 画像の幅と高さを書き込む
                        f.write(info)  # オブジェクトの情報を書き込む

                    count += 1     
        #classes.txtを作る
        info = """shinkansen
densya"""

        text_save_path = os.path.join(output_folder_path, "classes.txt")
        with open(text_save_path, "w") as f:
            f.write(info)
        
                        

    #def annotation(self, input_folder_path, output_folder_path):
                         
    def select_train_type(self, num):
        if(num == 0):
            train_type = "shinkansen"
            number = num
            return train_type, number 
        elif(num == 1):
            train_type = "densya"
            number = num
            return train_type, number
        
    