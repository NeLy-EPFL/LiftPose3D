'''
Converts binary files into grayscale images produced by basler cameras.
Uses memmory mapped files in case binary files cannot fit into the memory.

https://www.baslerweb.com/en/
'''
import numpy as np
import glob
import os
import cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', type=str)
parser.add_argument('--recursive', default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

#image_shape = [480,960]
image_shape = [700, 1792]
image_size = image_shape[0]*image_shape[1]

if __name__=='__main__':
    filename_list = list(glob.iglob('{}**/*.tmp'.format(args.path), recursive=args.recursive))
    print("Files Found : ", filename_list)
    for filename in filename_list:
        try:
            print("Processing: {}".format(filename))
            path = os.path.dirname(filename)

            print("Reading file: ", filename)
            d = np.memmap(filename, dtype=np.uint8, mode='r')
            print("File shape", d.shape)
            cam_id = int(filename[-5])
            num_images = int(d.shape[0]/image_size)
            print ("Num images: ", num_images)
            for img_id in tqdm(range(num_images)):
                img = d[img_id*image_size:(img_id+1)*image_size].reshape(image_shape)
                image_name = "camera_{}_img_{:006d}.jpg".format(cam_id, img_id)
                image_path = os.path.join(path, image_name)

                cv2.imwrite(image_path, img)
        except BaseException as e:
            print("Cannot process {}".format(filename))
            print(e)
            continue
