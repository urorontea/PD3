import project_processing01 as pp

from TrainValTest_2 import devide_img
from CombineTwoFolders import combine,combine01


#これから以下のフォルダにいっぱい画像を保存するぞ
dataset_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/"

#引数のtypeには電車の形式番号を入れる（例）521系→type = 521
def save_densya(youtube_url, dataset_folder_path, save_sum, type):
  
  #画像を保存する場所を指定
  save_img_dir = dataset_folder_path + type + "/"
  #動画から画像を保存する場所
  input_folder_path = save_img_dir
  #保存した画像で物体認識をした後に保存する場所
  output_folder_path = save_img_dir + "cropped/"
  pp_instance_den = pp.project_processing()
  video_title = pp_instance_den.get_video_title(youtube_url)

  #select_train_type(num) →  return train_type, number
  train_type =  pp_instance_den.select_train_type(1)[0]
  number = pp_instance_den.select_train_type(1)[1]
  
  #ダウンロードした動画の場所
  video_path = save_img_dir + video_title + ".mp4"

  pp_instance_den.get_random_frame(train_type, youtube_url, save_img_dir, save_sum)       
  pp_instance_den.croping(train_type, number, input_folder_path, output_folder_path)
  print("おしまい")

def save_shinkansen(youtube_url, dataset_folder_path, save_sum, type):
  
  #画像を保存する場所を指定
  save_img_dir = dataset_folder_path + type + "/"
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

pp_instance_den = pp.project_processing()
pp_instance_shin = pp.project_processing()


#nの値によって保存する電車を変える。０の時は保存しない
n = 0
if n == 1:
    #313系
    youtube_url = "https://www.youtube.com/watch?v=SP0oAn8sQoU"
    type = "313"
elif n == 2:
    #415系
    youtube_url = "https://www.youtube.com/watch?v=EPn430ztgiA"
    type = "415"
elif n == 3:
    #521系
    youtube_url = "https://www.youtube.com/watch?v=BMEmhlf13aU"
    type = "521"
elif n == 4:
    #W7系
    youtube_url = "https://www.youtube.com/watch?v=ll0DX9DrdNk"
    type = "W7"
elif n == 5:
    #こまち、はやぶさ
    youtube_url = "https://www.youtube.com/watch?v=qJmdUDJpFfA"
    type = "E5&E6"



save_sum = 400
#save_densya(youtube_url, dataset_folder_path, save_sum, type)
#save_shinkansen(youtube_url, dataset_folder_path, save_sum, type)

#フォルダ合体（二つを1つに）
p = 0
if(p == 1):
    folder1_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/E5&E6/cropped"
    folder2_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/W7/cropped"
    output_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/shin"
    pp_instance_shin.combine(folder1_path, folder2_path, output_folder_path)
elif(p == 2):
    folder1_path =  "/home/nozaki/ML/ultralytics/pic/dataset0624/521/cropped"
    folder2_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/415/cropped"
    output_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/den"
    pp_instance_den.combine(folder1_path, folder2_path, output_folder_path)

#フォルダ合体
f = 1
if(f == 1):
    folder1_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/521/cropped"
    output_folder_path = "/home/nozaki/ML/ultralytics/pic/dataset0624/den"
    count = pp_instance_den.count_images(output_folder_path)
    pp_instance_den.combine01(folder1_path, output_folder_path, count)
