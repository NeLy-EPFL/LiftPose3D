import cv2
import sys
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]

    img_names = [f for f in listdir(imgs_dir+"top_view/")\
        if isfile(join(imgs_dir+"top_view/", f)) and f.endswith(".jpg")]
    for idx in range(len(img_names)):
        img = cv2.imread(imgs_dir + "top_view/" + img_names[idx])
        cv2.imshow("top", img)
        cv2.waitKey(5)
    cv2.destroyWindow("top")

    img_names = [f for f in listdir(imgs_dir+"side_view/")\
        if isfile(join(imgs_dir, f)) and f.endswith(".jpg")]
    for img_name in img_names:
        img = cv2.imread(imgs_dir + "side_view/" + img_name)
        cv2.imshow("side", img)
        cv2.waitKey(5)
    cv2.destroyWindow("side")
