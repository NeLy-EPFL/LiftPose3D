from os import listdir, system
from os.path import isfile, isdir, join
import random
random.seed(7)
import sys
from tqdm import tqdm
import cv2
import skvideo.io

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
    img_names_side = [f for f in listdir(side_dir) if isfile(join(side_dir, f)) and f.endswith(".jpg")]
    img_names_top.sort()
    img_names_side.sort()

    pick_every = int(len(img_names_top) / 100)
    i = 1
    tobelab_top = []
    tobelab_side = []
    for (name_top, name_side) in zip(img_names_top, img_names_side):
        if i % pick_every == 0:
            tobelab_top.append(name_top)
            tobelab_side.append(name_side)
            print(name_top, name_side)
        i += 1
    if len(tobelab_top) > 100:
        tobelab_top = tobelab_top[:100]
        tobelab_side = tobelab_side[:100]
    while len(tobelab_top) < 100:
        (name_top, name_side) = random.choice(list(zip(img_names_top, img_names_side)))
        tobelab_top.append(name_top)
        tobelab_side.append(name_side)
    print(len(tobelab_top))

    img_t = cv2.imread(top_dir + tobelab_top[0], cv2.IMREAD_GRAYSCALE)
    fps = 60
    width = img_t.shape[1]
    height = img_t.shape[0]
    crf = 17
    writer_top = skvideo.io.FFmpegWriter(top_dir +"tobelab_top_video" + post_top +".mp4",
        inputdict={'-s':'{}x{}'.format(width,height)},
        outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf),
            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )
    for img_nt in tobelab_top:
        img_t = cv2.imread(top_dir + img_nt, cv2.IMREAD_GRAYSCALE)
        writer_top.writeFrame(img_t)
    writer_top.close()

    img_t = cv2.imread(side_dir + tobelab_side[0], cv2.IMREAD_GRAYSCALE)
    fps = 60
    width = img_t.shape[1]
    height = img_t.shape[0]
    crf = 17
    writer_side = skvideo.io.FFmpegWriter(side_dir +"tobelab_side_video" + post_side +".mp4",
        inputdict={'-s':'{}x{}'.format(width,height)},
        outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf),
            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )
    for img_nt in tobelab_side:
        img_t = cv2.imread(side_dir + img_nt, cv2.IMREAD_GRAYSCALE)
        writer_side.writeFrame(img_t)
    writer_side.close()
