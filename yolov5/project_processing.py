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
        
    #youtube動画をダウンロードする
    def dlyt(self, video_url):
        title = self.get_video_title(video_url)
        ydl_opts = {
              'format': 'bestvideo[ext=mp4]+noaudio[ext=m4a]/mp4',
              'outtmpl': '{}'.format(title +'.mp4'),
        }
        
        #動画を保存するディレクトリのパス
        video_path = f'home/nozaki/ML/ultralytics/pic/sozai/'

        #事前に動画をvideo_fileに読み込む
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info_dict = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info_dict)
            #print(current_dir+'/daunro_dositayatu.mp4')
            #cap = cv2.VideoCapture(current_dir+'/' + dl_name +'.mp4')
            cap = cv2.VideoCapture(video_path + filename)
            #print(f"cap = {cap}")
            
            return cap


    #動画(mp4のファイル)からランダムで1枚画像を取って指定されたディレクトリに保存する
    def get_random_frame(self, name, video_path, save_img_dir, img_num):
        
        #cap = self.dlyt(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #frame_count = 2
        print(f"frame_count = {frame_count}")

        #引数にて指定された回数ランダムで動画のシーンを切り取る
        for i in range(img_num):
            name_1 = name +'_'+ str(i).zfill(4) + '.jpg'  #シーンの名前を拡張子込みでつける
            file_path = os.path.join(save_img_dir, name_1) #ファイルのフルパスをつくる
            os.makedirs(save_img_dir, exist_ok=True) #ディレクトリが無ければ作る

            random_frame_number = random.randint(0, frame_count-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(file_path, frame) #成功時ファイルの書き込み
                print(f"Saved random frame {random_frame_number} as random_frame.jpg")
            else:
                print(f"Failed to read frame {random_frame_number}") #失敗時

    #ランダムで与えられた動画のフレームを取るやつ
    #保存されたファイルではなくVideoCaptureオブジェクトを引数にとるver
    #返り値もnp.ndarray形式
    def get_random_frame_1(cap,frame_count):

        random_frame_number = random.randint(0, frame_count-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        ret, frame = cap.read()
        if ret:
            print(f"Returned randomframe successfully")
            return frame #成功時ndarrayを返す

        else:
            print(f"Failed to read frame {random_frame_number}") #失敗時
            return 0
        


    def croping(img, output_folder_path, identifier, numbering, model):
       
        #画像として保存するよーん
        #img:加工させる画像、output_folder_path:出力するフォルダのパス
        #identifier:識別するものの名前、numbering:imgのナンバリング
        # 入力されたモデルで物体検出を実行する
        result = model(img)

         # 物体検出結果をPandasのDataFrame形式で取得する
        obj = result.pandas().xyxy[0]
        count = 0
        # 検出された電車のバウンディングボックスの中身を別のフォルダに保存する
        for j in range(len(obj)):
            name = obj.name[j]
            xmin = obj.xmin[j]
            ymin = obj.ymin[j]
            xmax = obj.xmax[j]
            ymax = obj.ymax[j]

            # 電車として認識されたモノだけ保存する
            if name == identifier:
                # バウンディングボックスの情報を元に、元画像から切り抜く領域を決定する
                cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

                # 保存先のパスを生成する
                #filename = f'train_{str(i).zfill(4)}_{str(j).zfill(4)}.jpg'
                filename = f'train_{str(numbering).zfill(4)}.jpg'
                save_path = os.path.join(output_folder_path, filename)

                # 切り抜いた画像を保存する
                cv2.imwrite(save_path, cropped_img)

                print(f'Saved {filename}')
                count = 1
            return count


        # バウンディングボックスを描画する
        '''
        result.render()
        cv2_imshow(result.ims[0])
        cv2.waitKey(0)
        '''
    def croping_1(img, identifier, numbering, model):
    #img:加工させる画像、output_folder_path:出力するフォルダのパス
    #identifier:識別するものの名前、numbering:imgのナンバリング
    #こっちはファイルとして保存しないタイプ

    # 入力されたモデルで物体検出を実行する
        result = model(img)
    
    # 物体検出結果をPandasのDataFrame形式で取得する
        obj = result.pandas().xyxy[0]
        count = 0
        cropped_img = None
    # 検出された電車のバウンディングボックスの中身を別のフォルダに格納して返す
        for j in range(len(obj)):
            name = obj.name[j]
            xmin = obj.xmin[j]
            ymin = obj.ymin[j]
            xmax = obj.xmax[j]
            ymax = obj.ymax[j]
        
        # 電車として認識されたモノだけ返す
            if name == identifier:
            # バウンディングボックスの情報を元に、元画像から切り抜く領域を決定する
                cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

                count = 1

        return count, cropped_img



    
    def croping_2(img, identifier, numbering, model):
        #こっちはファイルとして保存しないタイプ

        #img:加工させる画像、output_folder_path:出力するフォルダのパス
        #identifier:識別するものの名前、numbering:imgのナンバリング
    
         # 入力されたモデルで物体検出を実行する
        result = model(img)
    
         # 物体検出結果をPandasのDataFrame形式で取得する
        obj = result.pandas().xyxy[0]
        count = 0
        cropped_img = None
        # 検出された電車のバウンディングボックスの中身を別のフォルダに格納して返す
        for j in range(len(obj)):
            name = obj.name[j]
            xmin = obj.xmin[j]
            ymin = obj.ymin[j]
            xmax = obj.xmax[j]
            ymax = obj.ymax[j]
        
        # 電車として認識されたモノだけ返す
            if name == identifier:
            # バウンディングボックスの情報を元に、元画像から切り抜く領域を決定する
                cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

                count = 1

        return count, cropped_img



   
    