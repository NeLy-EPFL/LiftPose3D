'''
Stitch videos into a montage seen in the paper by the following commants
in the terminal.
'''

#ffmpeg -i arena_marked.mp4 -i DLC_prediction.mp4 -filter_complex "[0:v]scale=-1:600[v0];[v0][1:v]hstack=inputs=2" output.mp4
#ffmpeg -i LiftPose3D_prediction.mp4 -i projected_traces.mp4 -filter_complex "[0:v]scale=-1:400[v0];[v0][1:v]hstack=inputs=2" output_2.mp4

#ffmpeg -i output.mp4 -i output_2.mp4 -filter_complex "[0:v]scale=650:-1,pad='iw+mod(iw\,2)':'ih+mod(ih\,2)'[v0];[v0][1:v]vstack=inputs=2" video_4.mp4