import project_processing01 as pp



#新幹線編スタート
youtube_url = "https://www.youtube.com/watch?v=A8n6uwJs4bA"
#youtubeの動画から適当な画像を保存する際に保存先
save_img_dir = "/home/nozaki/ML/ultralytics/pic/dataset0619_3/shin/"
#電車として認識されたモノだけを保存する
input_folder_path =  save_img_dir
output_folder_path = save_img_dir + "cropped/"

pp_instance_shin = pp.project_processing()
#タイトル取得
video_title = pp_instance_shin.get_video_title(youtube_url)

#画像の保存枚数
save_sum = 50

train_type =  pp_instance_shin.select_train_type(0)[0]
number = pp_instance_shin.select_train_type(0)[1] 

print(f"train_type = {train_type}")
#ダウンロードした動画の場所
video_path = save_img_dir + video_title + ".mp4"

pp_instance_shin.get_random_frame(train_type, youtube_url, save_img_dir, save_sum)           
pp_instance_shin.croping(train_type, number, input_folder_path, output_folder_path)




#電車編スタート
#電車の動画
youtube_url =  "https://www.youtube.com/watch?v=h44if9TrMeI"
#youtubeの動画から適当な画像を保存する際に保存先
save_img_dir = "/home/nozaki/ML/ultralytics/pic/dataset0619_3/den/"
#電車として認識されたモノだけを保存する
input_folder_path = save_img_dir
output_folder_path = save_img_dir + "cropped/"

pp_instance_den = pp.project_processing()
#タイトル取得
video_title = pp_instance_den.get_video_title(youtube_url)

#画像の保存枚数
save_sum = 50

#select_train_type(num) →  return train_type, number
train_type =  pp_instance_den.select_train_type(1)[0]
number = pp_instance_den.select_train_type(1)[1]
 
#ダウンロードした動画の場所
video_path = save_img_dir + video_title + ".mp4"

pp_instance_den.get_random_frame(train_type, youtube_url, save_img_dir, save_sum)       
pp_instance_den.croping(train_type, number, input_folder_path, output_folder_path)



