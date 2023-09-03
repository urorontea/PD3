import re
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


from TrainValTest_2 import devide_img
from CombineTwoFolders import combine


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
                #print(f"Saved random frame {random_frame_number} as {name_1}")
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

                #座標を０～１の間で表すぞ
                # 画像の幅と高さを取得
                image_width = image.shape[1]
                image_height = image.shape[0]

                # バウンディングボックスの座標をピクセル値から割合に変換
                xmin_normalized = xmin / image_width
                ymin_normalized = ymin / image_height
                xmax_normalized = xmax / image_width
                ymax_normalized = ymax / image_height

                     
                if name == "train" and score >=0.7:
                    #保存先のパスを生成
                    filename = f"{train_type}_{str(count).zfill(4)}"
                    img_filename = filename + ".jpg"
                    #print(f"img_filename = {img_filename}")
                    save_path = os.path.join(output_folder_path, img_filename)
                    #print(f"save_path = {save_path}")

                    #電車の画像を保存
                    #cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    
                    #cv2.rectangle(image, (int(xmin_normalized), int(ymin_normalized)), (int(xmax_normalized), int(ymax_normalized)), (0, 255, 0), 2)
                    cv2.imwrite(save_path, image)
                    #print(f"saved {img_filename}\n")

                    #物体認識をしてゲットした座標情報をテキストファイルに保存
                    info = f"{number} {xmin_normalized} {ymin_normalized} {xmax_normalized} {ymax_normalized}"
                    text_filename = filename + ".txt"
                    save_path = os.path.join(output_folder_path, text_filename)
                    #オブジェクトの情報を書き込む
                    with open(save_path, "w") as f:
                        #f.write(f"{cropped_img.shape[1]} {cropped_img.shape[0]}\n")  # 画像の幅と高さを書き込む
                        f.write(info)  # オブジェクトの情報を書き込む

                    count += 1 

        #classes.txtを作る
        info = """shinkansen\ndensya"""

        text_save_path = os.path.join(output_folder_path, "classes.txt")
        with open(text_save_path, "w") as f:
            f.write(info)

        #保存した画像を消す（物体認識をする前の画像を消す）
        # ディレクトリ内のファイルを取得
        file_list = os.listdir(input_folder_path)
        #print("今から画像を消すよ")
        #rint(file_list)
        #print(input_folder_path)

        for file_name in file_list:
            file_path = os.path.join(input_folder_path, file_name)

            # ファイルが存在し、拡張子が.jpgの場合にのみ削除
            if os.path.isfile(file_path) and file_name.endswith(".jpg"):
                os.remove(file_path)
                print(f"{file_name} を削除しました")
 
        
                        

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
        
    
    #二つのふぉるだを1つにしちゃうぞ
    def combine(self, folder1_path, folder2_path, output_folder_path):
        # 元のフォルダのパス
        #folder1_path = "/home/nozaki/ML/ultralytics/pic/dataset0619_4/shin/cropped/"
    # folder2_path = "/home/nozaki/ML/ultralytics/pic/dataset0619_4/den/cropped/"
        # 新しいフォルダのパス
    # output_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0619_4/aaaa"
        
        # 出力フォルダが存在しない場合は作成する
        os.makedirs(output_folder_path, exist_ok=True)

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
                    """

                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder1_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                     """

                    counter1 += 1
                    
                elif "shinkansen" in filename:
                    image_src = os.path.join(folder1_path, filename)
                    image_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.jpg")
                    shutil.copyfile(image_src, image_dst)
                    """
                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder1_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                    """

                    counter2 += 1

        # フォルダ2の画像とテキストファイルをコピー
        for filename in os.listdir(folder2_path):
            if filename.endswith(".jpg"):
                if "densya" in filename:
                    image_src = os.path.join(folder2_path, filename)
                    image_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.jpg")
                    shutil.copyfile(image_src, image_dst)
                    """

                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder2_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"densya_{counter1:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                     """

                    counter1 += 1
                   
                elif "shinkansen" in filename:
                    image_src = os.path.join(folder2_path, filename)
                    image_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.jpg")
                    shutil.copyfile(image_src, image_dst)
                    """

                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder2_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"shinkansen_{counter2:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                    """

                    counter2 += 1

        print("conbined")

    def combine01(self, folder1_path, output_folder_path, pic_sum):
        counter = pic_sum
        for filename in os.listdir(folder1_path):
            if filename.endswith(".jpg"):
                if "densya" in filename:
                    image_src = os.path.join(folder1_path, filename)
                    image_dst = os.path.join(output_folder_path, f"densya_{counter:04d}.jpg")
                    shutil.copyfile(image_src, image_dst)
                    """
                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder1_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"densya_{counter:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                    """
                    counter += 1

                elif "shinkansen" in filename:
                    image_src = os.path.join(folder1_path, filename)
                    image_dst = os.path.join(output_folder_path, f"shinkansen_{counter:04d}.jpg")
                    shutil.copyfile(image_src, image_dst)
                    """

                    txt_filename = filename.replace(".jpg", ".txt")
                    txt_src = os.path.join(folder1_path, txt_filename)
                    txt_dst = os.path.join(output_folder_path, f"shinkansen_{counter:04d}.txt")
                    shutil.copyfile(txt_src, txt_dst)
                    """

                    counter += 1
        print("conbined")
        
    #train:val:test に分けちゃうお
    def devide_img(self, data_folder):
        # フォルダのパスと分割比率の設定
        #data_folder = "/home/nozaki/ML/ultralytics/pic/dataset0619_4/aaaa/"
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        # データセット内のファイルリストを取得
        file_list = os.listdir(data_folder)
        file_list.sort()
        #print(file_list,'\n')

        # フォルダの作成
        train_folder = os.path.join(data_folder, "train")
        val_folder = os.path.join(data_folder, "val")
        test_folder = os.path.join(data_folder, "test")

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # データセットの分割
        densyaIMG_files = [file for file in file_list if "densya" in file and "jpg" in file]
        densyaIMG_files.sort()
        densyaTXT_files = [file for file in file_list if "densya" in file and "txt" in file]
        densyaTXT_files.sort()
        shinkansenIMG_files = [file for file in file_list if "shinkansen" in file and "jpg" in file]
        shinkansenIMG_files.sort()
        shinkansenTXT_files = [file for file in file_list if "shinkansen" in file and "txt" in file]
        shinkansenTXT_files.sort()

        #random.shuffle(densya_files)
        #random.shuffle(shinkansen_files)

        num_densya = len(densyaIMG_files)
        num_shinkansen = len(shinkansenIMG_files)

        num_train_densya = int(num_densya * train_ratio)
        num_val_densya = int(num_densya * val_ratio)
        num_test_densya = num_densya - num_train_densya - num_val_densya

        num_train_shinkansen = int(num_shinkansen * train_ratio)
        num_val_shinkansen = int(num_shinkansen * val_ratio)
        num_test_shinkansen = num_shinkansen - num_train_shinkansen - num_val_shinkansen

        # 電車データの分割
        for i in range(num_densya):
            img_filename = densyaIMG_files[i]
            #print("img_filenameは、",img_filename)
            txt_filename = img_filename.replace(".jpg", ".txt")
            #print("txt_filenameは、",txt_filename)
            
            # ファイルのパスを取得
            img_path = os.path.join(data_folder, img_filename)
            txt_path = os.path.join(data_folder, txt_filename)
            
            # 分割先のフォルダを選択
            if i < num_train_densya:
                #dest_folder = os.path.join(train_folder, "densya")
                dest_folder = train_folder
            elif i < num_train_densya + num_val_densya:
                #dest_folder = os.path.join(val_folder, "densya")
                dest_folder = val_folder
            else:
                #dest_folder = os.path.join(test_folder, "densya")
                dest_folder = test_folder
            
            # ファイルの移動
            shutil.move(img_path, os.path.join(dest_folder, img_filename))
            #print(img_filename,"を移動しました")
            shutil.move(txt_path, os.path.join(dest_folder, txt_filename))

        # 新幹線データの分割
        for i in range(num_shinkansen):
            img_filename = shinkansenIMG_files[i]
            txt_filename = img_filename.replace(".jpg", ".txt")
            
            # ファイルのパスを取得
            img_path = os.path.join(data_folder, img_filename)
            txt_path = os.path.join(data_folder, txt_filename)
            
            # 分割先のフォルダを選択
            if i < num_train_shinkansen:
                #dest_folder = os.path.join(train_folder, "shinkansen")
                dest_folder = train_folder
            elif i < num_train_shinkansen + num_val_shinkansen:
                #dest_folder = os.path.join(val_folder, "shinkansen")
                dest_folder = val_folder
            else:
                #dest_folder = os.path.join(test_folder, "shinkansen")
                dest_folder = test_folder

            # ファイルの移動
            shutil.move(img_path, os.path.join(dest_folder, img_filename))
            shutil.move(txt_path, os.path.join(dest_folder, txt_filename))

        print("some files is made")
            
    #dataset.yamlを作っちゃうぞ
    def make_yaml(self, input_folder_path):
        save_path = input_folder_path + "dataset.yaml"
        with open(save_path, "w") as f:
            f.write(f"train: {input_folder_path}train/\n")
            f.write(f"val: {input_folder_path}val/\n")
            f.write(f"test: {input_folder_path}test/\n")
            f.write("nc: 2\n")
            f.write("names: ['shinkansen', 'densya']")
        print("dataset.yaml is made")

    def make_classes(self, input_folder_path):
        text_save_path = os.path.join(input_folder_path, "classes.txt")
        with open(text_save_path, "w") as f:
            f.write("""shinkansen\ndensya""")
        copy_test = input_folder_path + "test/"
        copy_train = input_folder_path + "train/"
        copy_val = input_folder_path + "val/"
        shutil.copyfile(text_save_path, copy_test + "classes.txt")
        shutil.copyfile(text_save_path, copy_train + "classes.txt")
        shutil.copyfile(text_save_path, copy_val + "classes.txt")
        print("make_classes")

    def resize_sq(self,input_folder_path):

        #new_fol='new_pic'
        #新しいフォルダを作成
        #os.makedirs(new_fol, exist_ok=True)
        #画像ファイル一覧を取得
        image_files = glob.glob(os.path.join(input_folder_path, '*.jpg')) 
    
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

    def count_images(self, folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # 画像ファイルの拡張子リスト
        count = 0

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    count += 1

        return count




