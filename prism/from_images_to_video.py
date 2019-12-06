import cv2
import skvideo.io
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

DEBUG = False

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]
    if not imgs_dir.endswith("/") : imgs_dir += "/"

    imgs_dir_spl = imgs_dir.split("/")
    post_top = "_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6] +"_"+ imgs_dir_spl[7]
    post_side = "_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6] +"_"+ imgs_dir_spl[7]
    top_dir = "top_view" + post_top +"/"
    side_dir = "side_view" + post_side +"/"

    print("[*] converting images from\n"+top_dir+"\n"+side_dir)

    img_names_top = [f for f in listdir(imgs_dir + top_dir)\
        if isfile(join(imgs_dir + top_dir, f)) and f.endswith(".jpg")]
    img_names_side = [f for f in listdir(imgs_dir + side_dir)\
        if isfile(join(imgs_dir + side_dir, f)) and f.endswith(".jpg")]
    img_names_top.sort()
    img_names_side.sort()
    
    img_t = cv2.imread(imgs_dir + top_dir + img_names_top[0], cv2.IMREAD_GRAYSCALE)
    fps = 60
    width = img_t.shape[1]
    height = img_t.shape[0]
    crf = 17
    writer_top = skvideo.io.FFmpegWriter(imgs_dir + top_dir +"top_video" + post_top +".mp4", 
        inputdict={'-s':'{}x{}'.format(width,height)},
        outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf),
            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for idx in tqdm(range(len(img_names_top))):
        img_nt = img_names_top[idx]
        
        img_t = cv2.imread(imgs_dir + top_dir + img_nt, cv2.IMREAD_GRAYSCALE)
        if DEBUG:
            cv2.imshow("top", img_t)
            cv2.waitKey(10)
        writer_top.writeFrame(img_t)

    writer_top.close()

    img_s = cv2.imread(imgs_dir + side_dir + img_names_side[0], cv2.IMREAD_GRAYSCALE)
    fps = 60
    width = img_s.shape[1]
    height = img_s.shape[0]
    crf = 17
    writer_side = skvideo.io.FFmpegWriter(imgs_dir + side_dir +"side_video" + post_side +".mp4",
        inputdict={'-s':'{}x{}'.format(width,height)},
        outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf),
            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for idx in tqdm(range(len(img_names_side))):
        img_ns = img_names_side[idx]

        img_s = cv2.imread(imgs_dir + side_dir + img_ns, cv2.IMREAD_GRAYSCALE)
        if DEBUG:
            cv2.imshow("top", img_s)
            cv2.waitKey(10)
        writer_side.writeFrame(img_s)

    writer_side.close()
