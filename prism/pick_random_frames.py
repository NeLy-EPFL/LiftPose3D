from os import listdir
from os.path import isfile, isdir, join
import random
import sys
from tqdm import tqdm

args = sys.argv
data_dir = args[1]
if not data_dir.endswith("/") : data_dir += "/"

imgs_dir_spl = data_dir.split("/")
post_top = "_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
post_side = "_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]

top_dir = data_dir+"top_view"+ post_top +"/"
side_dir = data_dir+"side_view"+ post_side +"/"
topvideo_dir = "top_video"+ post_top +"/"
sidevideo_dir = "side_video"+ post_side +"/"

if __name__ == '__main__':
    print(f"\n[*] reading images name from {data_dir:s}")
    img_names_top = [f for f in listdir(top_dir) if isfile(join(top_dir, f)) and f.endswith(".jpg")]
    img_names_side = [f for f in listdir(top_dir) if isfile(join(top_dir, f)) and f.endswith(".jpg")]
    img_names_top.sort()
    img_names_side.sort()
    
    for (name_top, name_side) in random.sample(zip(img_names_top, img_names_side), 100):
        '''
        os.system(
            "cp "top_dir+name_top+" ./jointTracking-PrismData-2019-12-06/labeled-data/"+\
            topvideo_dir)
        os.system(
            "cp "side_dir+name_side+" ./jointTracking-PrismData-2019-12-06/labeled-data/"+\
            sidevideo_dir)
        '''

        print(name_top, name_side)
