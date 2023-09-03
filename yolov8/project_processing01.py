#from email.mime import image
import re
#from symbol import typelist
from ultralytics import YOLO
import cv2
import random
import os
import yt_dlp
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import shutil
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
            'outtmpl': 'movie/%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info_dict)

        video_path = filename  # ダウンロードされたファイルのパスを使用

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count)

        for i in range(img_num):
            #name_1 = name + '_' + str(i) + '.png'
            name_new = f'{name}_{str(i).zfill(3)}.png'
            #filename = f'train_{str(i).zfill(4)}.png'

            file_path = os.path.join(dir_path, name_new)

            #出力フォルダが存在しない場合は作成する
            os.makedirs(dir_path, exist_ok=True)
            random_frame_number = random.randint(0, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(file_path, frame)
                print(f"Saved random frame {random_frame_number} as {name_new}")
            else:
                print(f"Failed to read frame {random_frame_number}")
        print(f"saved at {dir_path}")


   
    def croping(self, train_type, input_folder_path, output_folder_path):
        
        #モデルをロードする
        #model = torch.hub.load("/home/ultralytics", "yolov8n.pt", pretrained=True)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        # 画像として保存するよーん
        # img:加工させる画像、output_folder_path:出力するフォルダのパス
        # name:識別するものの名前、
        # 入力されたモデルで物体検出を実行する
        # 拡張子が.pngの画像ファイルを取得
        image_files = glob.glob(os.path.join(input_folder_path, '*.png')) 

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
                score = obj.confidence[j]
                     
                if name == "train" and score >=0.9:
                    #保存先のパスを生成
                    filename = f"{train_type}_{str(count).zfill(3)}"
                    img_filename = filename + ".png"
                    save_path = os.path.join(output_folder_path, img_filename)
            
                    cv2.imwrite(save_path, image)
                    print(f"saved {img_filename}\n")

                    count += 1 

        #保存した画像を消す（物体認識をする前の画像を消す）
        # ディレクトリ内のファイルを取得
        file_list = os.listdir(input_folder_path)

        for file_name in file_list:
            file_path = os.path.join(input_folder_path, file_name)

            # ファイルが存在し、拡張子が.pngの場合にのみ削除
            if os.path.isfile(file_path) and file_name.endswith(".png"):
                os.remove(file_path)
                print(f"{file_name} を削除しました")
        #物体認識をする前の画像フォルダを削除する
        os.rmdir(input_folder_path)
        print(f"{input_folder_path} を削除しました")
 
 
    def resize_sq(self,input_folder_path):

        #new_fol='new_pic'
        #新しいフォルダを作成
        #os.makedirs(new_fol, exist_ok=True)
        #画像ファイル一覧を取得
        image_files = glob.glob(os.path.join(input_folder_path, '*.png')) 
    
        #調整後サイズを指定(横幅、縦高さ)
        size=(900,900)
        #リサイズ処理開始
        for image_file in image_files:
            base_pic=np.zeros((size[1],size[0],3),np.uint8)
            pic1=cv2.imread(image_file,cv2.IMREAD_COLOR)
            h,w=pic1.shape[:2]
            ash=size[1]/h
            asw=size[0]/w
            if asw<ash:
                sizeas=(int(w*asw),int(h*asw))
            else:
                sizeas=(int(w*ash),int(h*ash))
            pic1 = cv2.resize(pic1,dsize=sizeas)
            base_pic[int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
            int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2),:]=pic1
            cv2.imwrite(input_folder_path + image_file, base_pic)
        print("resizedddd") 
