#to create video_1.mp4, execute these commands in sequence in the folder where the videos are

#stack videos vertically (call this in the folder where the images are)
#ffmpeg -i camera_1.mp4 -i camera_5.mp4 -filter_complex vstack=inputs=2 stacked.mp4

#annotate
#ffmpeg -i stacked.mp4 -vf drawtext="fontfile=/home/gosztolai/Dropbox/github/fly_data_analysis/Liftfly3D/DF3D/helvetica.ttf: text='Camera 2': fontcolor=white: fontsize=80: x=1*(w-text_w)/10: y=1*(h-text_h)/10/2" -codec:a copy stacked_b.mp4
#ffmpeg -i stacked_b.mp4 -vf drawtext="fontfile=/home/gosztolai/Dropbox/github/fly_data_analysis/Liftfly3D/DF3D/helvetica.ttf: text='Camera 5': fontcolor=white: fontsize=80: x=1*(w-text_w)/10: y=6*(h-text_h)/10" -codec:a copy stacked_2b.mp4

#stack videos horizontally (call this in the folder where the images are)
#ffmpeg -i stacked_2b.mp4 -i LiftPose3D_prediction.mp4 -filter_complex "[0:v]scale=-1:480[v0];[v0][1:v]hstack=inputs=2" video_1.mp4
