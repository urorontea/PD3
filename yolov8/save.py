from email.mime import base
from lib2to3.pytree import BasePattern
import project_processing01 as pp



#これから以下のフォルダにいっぱい画像を保存するぞ
base_path = "/home/nozaki/ultralytics/datasets/densya_type/"
#下記の電車を分類していくぞ
#type_list = ["207", "221", "223", "225", "227", "323", "521"]
#type_list = ["225", "227", "521"]
#引数のtypeには電車の形式番号を入れる（例）521系→type = 521
def save_densya(youtube_url, base_path, save_sum, type):
  
  #画像を保存する場所を指定
  save_img_dir = base_path + type + "before/"
  #動画から画像を保存する場所
  input_folder_path = save_img_dir
  #保存した画像で物体認識をした後に保存する場所
  output_folder_path = base_path + type
  pp_instance_den = pp.project_processing()
  #video_title = pp_instance_den.get_video_title(youtube_url)

  #select_train_type(num) →  return train_type, number
  #rain_type =  pp_instance_den.select_train_type(type_list, type)[0]
  #number = pp_instance_den.select_train_type(type_list, type)[1]
  
  #ダウンロードした動画の場所
  #video_path = save_img_dir + video_title + ".mp4"

  pp_instance_den.get_random_frame(type, youtube_url, save_img_dir, save_sum)       
  pp_instance_den.croping(type, input_folder_path, output_folder_path)
  print("おしまい")
"""
def save_shinkansen(youtube_url, base_path, save_sum, type):
  
  #画像を保存する場所を指定
  save_img_dir = base_path + type + "/"
  #動画から画像を保存する場所
  input_folder_path = save_img_dir
  #保存した画像で物体認識をした後に保存する場所
  output_folder_path = save_img_dir + "cropped/"
  pp_instance_shin = pp.project_processing()
  video_title =pp_instance_shin.get_video_title(youtube_url)

  #select_train_type(num) →  return train_type, number
  train_type =  pp_instance_shin.select_train_type(0)[0]
  number = pp_instance_shin.select_train_type(0)[1]
  
  #ダウンロードした動画の場所
  video_path = save_img_dir + video_title + ".mp4"
  #動画から画像をランダムに保存しちゃうぞ
  pp_instance_shin.get_random_frame(train_type, youtube_url, save_img_dir, save_sum)       
  #保存した画像で物体認識しちゃうぞ
  pp_instance_shin.croping(train_type, number, input_folder_path, output_folder_path)
  print("おしまい")
"""

pp_instance_den = pp.project_processing()
pp_instance_shin = pp.project_processing()


#画像の保存枚数は400枚とする(根拠はない)
save_sum = 500
#nの値によって保存する電車を変える。０の時は保存しない
#電車は一桁、新幹線は3桁
n = 8

if n == 1:
    #313系
    youtube_url = "https://www.youtube.com/watch?v=SP0oAn8sQoU"
    type = "313"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 2:
    #415系
    youtube_url = "https://www.youtube.com/watch?v=EPn430ztgiA"
    type = "415"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 3:
    #521系
    youtube_url = "https://www.youtube.com/watch?v=BMEmhlf13aU"
    type = "521"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 4:
    #225系
    youtube_url = "https://www.youtube.com/watch?v=MJFVBIhiGoI"
    type = "225"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 5:
    #227系
    youtube_url = "https://www.youtube.com/watch?v=7y2irpCNxhM"
    type = "227"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 6:
    #221系
    youtube_url = "https://www.youtube.com/watch?v=WKmIcpYH34w"
    type = "221"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 7:
    #223系
    youtube_url = "https://www.youtube.com/watch?v=UKdRzHTdRNM"
    type = "223"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 8:
    #207系
    youtube_url = "https://www.youtube.com/watch?v=KurB7vxFUQY"
    type = "207"
    save_densya(youtube_url, base_path, save_sum, type)
elif n == 9:
    #323系
    youtube_url = "https://www.youtube.com/watch?v=BQ_0m_DVypc"
    type = "323"
    save_densya(youtube_url, base_path, save_sum, type)








"""

elif n == 101:
    #W7系
    youtube_url = "https://www.youtube.com/watch?v=ll0DX9DrdNk"
    type = "W7"
    save_shinkansen(youtube_url, base_path, save_sum, type)
elif n == 102:
    #こまち
    youtube_url = "https://www.youtube.com/watch?v=Cr88elZQkyU"
    type = "E6"
    save_shinkansen(youtube_url, base_path, save_sum, type)
elif n == 103:
    #E5
    youtube_url = "https://www.youtube.com/watch?v=uHxp_2tJDcw"
    type = "E5"
    save_shinkansen(youtube_url, base_path, save_sum, type)

#save_densya(youtube_url, base_path, save_sum, type)
#save_shinkansen(youtube_url, base_path, save_sum, type)

#フォルダ合体（二つを1つに）
p = 0
#
if(p == 1):
    #新幹線と電車があるフォルダをまとめる
    folder1_path = base_path + "shin/"
    folder2_path = base_path + "den/"
    output_folder_path =base_path + "data/"
    pp_instance_shin.combine(folder1_path, folder2_path, output_folder_path)
elif(p == 2):
    #電車と新幹線をそれぞれ一つずつにまとめる
    folder1_path = base_path + "225/cropped"
    folder2_path = base_path + "227/cropped"
    #output_folder_path = base_path + "shin"
    output_folder_path = base_path + "data"
    pp_instance_den.combine(folder1_path, folder2_path, output_folder_path)

#pp_instance_den.make_yaml("/home/nozaki/ML/ultralytics/pic/dataset0625/")

#フォルダ合体
f = 0
if(f == 1):
    folder1_path = base_path + "313/cropped/"
    output_folder_path = base_path + "data"
    count = pp_instance_den.count_images(output_folder_path)
    pp_instance_den.combine01(folder1_path, output_folder_path, count)

#train:val:testに分ける

t = 0
if(t == 1):
    data_folder = base_path + "data/"
    pp_instance_den.devide_img(data_folder)

#pp_instance_den.make_yaml(base_path + "data/")
#type_list = ["225", "227", "521"]
destination_path = base_path + "data/"
cp = 0
if(cp == 1):
    pp_instance_den.copy_file(base_path, type_list, destination_path)
"""