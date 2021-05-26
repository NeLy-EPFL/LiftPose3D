import os
import cv2 as cv
import numpy as np
from scipy import ndimage
from pathlib import Path

vid_paths = ['/home/lobato/Desktop/liftPose_data/videos/liftPose_station1/liftPose_station1_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station5/liftPose_station5_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station6/liftPose_station6_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station7/liftPose_station7_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station8/liftPose_station8_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station10/liftPose_station10_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station12/liftPose_station12_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station13/liftPose_station13_80fps.mp4',
             '/home/lobato/Desktop/liftPose_data/videos/liftPose_station17/liftPose_station17_80fps.mp4']
             
cropSize=290

def crop_img(frame, coords, cropSize):
    """
    Crop square around a selected point, with area given by the cropSize parameter.

    Args:
        x, y: centroid of the crop
        frame
        cropSize:      

    Returns:
        A frame/numpy array/image contaning the cropped animal from the selected frame.
    """
    x, y = int(coords[0]), int(coords[1])
    
    x1 = x - cropSize//2 if x - cropSize//2 > 0 else 0
    x2 = x + cropSize//2 if x + cropSize//2 < frame.shape[1] - 1 else frame.shape[1] - 1
    y1 = y - cropSize//2 if y - cropSize//2 > 0 else 0
    y2 = y + cropSize//2 if y + cropSize//2 < frame.shape[0] - 1 else frame.shape[0] - 1
    #print(frame.shape, x, y,x1,x2,y1,y2)
    cropped_frame = frame[y1:y2, x1:x2].copy()
    
    return cropped_frame

def get_crop(img, coords, cropSize):
    crop = np.zeros_like(img[0:cropSize, 0:cropSize])    
    cropped_frame = crop_img(img, coords, cropSize)
    crop[0:cropped_frame.shape[0],0:cropped_frame.shape[1]] = cropped_frame
    
    return crop

def remove_background(img):
    img_blur = cv.medianBlur(img,7) 
    lower = np.array ([0,50,20])
    upper = np.array ([100,255,255])
    img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_blur, lower, upper)
    
    fly = cv.bitwise_and(img_blur, img_blur, mask=mask)
    #gray = cv.cvtColor(fly, cv.COLOR_BGR2GRAY)
    gray = fly[:,:,2]

    return gray

def get_closer_component(img,coord):
    output = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
    stats = np.transpose(output[2])
    sizes = stats[4]
    h,w = img.shape
    closer_to =np.array([h,w])*coord

    label=-1
    for i, (a,b) in enumerate(zip(stats[0],stats[3])):
        if closer_to[1] > a and closer_to[1]<a+b:
            label = i
    
    closerComponent = np.zeros_like(img)
    if label > -1:
        closerComponent[np.where(output[1] == label)] = 255

    return closerComponent
    

def check_angle(img, angle):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _,th = cv.threshold(gray,175,255,cv.THRESH_BINARY_INV)
    h, w = th.shape
    roi = h//4    
    top = th[0:roi, :]
    bottom = th[h-roi:h, :]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,10))

    top_open = cv.erode(top,kernel,iterations = 1)
    bottom_open = cv.erode(bottom,kernel,iterations = 1)

    head = get_closer_component(top_open,np.array([1,0.5]))
    wings = get_closer_component(bottom_open,np.array([0,0.5]))

    if np.sum(head) > np.sum(wings):
        angle = (angle+180)%360

    #cv.imshow('gray',th)
    #cv.imshow('top', head)
    #cv.imshow('bottom',wings)
    #cv.waitKey()
    
    return angle

def main():
    for vid_path in vid_paths:
        print('Experiment:'+vid_path)
        # Name of source video and paths
        out_path = vid_path.replace('.mp4', '_flyCrop.mp4')
        # Open video
        cap = cv.VideoCapture(vid_path)
        if cap.isOpened() == False:
            raise Exception('Video file cannot be read! Please check in_path to ensure it is correctly pointing to the video file')

        angles=[]
        raw_imgs=[]
        rot_imgs=[]
        count_flip = 0
        fps = int(cap.get(5))

        print('Reading video...')
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                raw_imgs.append(frame)
            else:
                break

        for frame_number, frame in enumerate(raw_imgs):
            print('Cropping frame:' + str(frame_number+1) +'/'+str(len(raw_imgs)), end='\r')
        #for i in range(1,1801):
            #img_name = 'img_'+str(i).zfill(3)+'.jpg'
            #img_path = os.path.join(folder,img_name)
            #img = cv.imread(img_path)
            gray = remove_background(frame)
            _,th = cv.threshold(gray,10,255,cv.THRESH_BINARY)

            contour, _ = cv.findContours(th, 1, 2)
            contour = max(contour, key=cv.contourArea)
            ellipse = cv.fitEllipse(contour)   
            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            ellipse_angle = ellipse[2]    
            angle = ellipse_angle
            if angles:
                diff_ellipse = angles[-1]-ellipse_angle

                if abs(diff_ellipse)>160:
                    angle = ellipse_angle + 180*(diff_ellipse/abs(diff_ellipse))
                    diff_orientation = angles[-1]-angle
                    if abs(diff_orientation)>160:
                        angle=ellipse_angle

            angles.append(angle)
            crop = get_crop(frame,[cx,cy],cropSize)
            rot_fly = ndimage.rotate(crop, angle, reshape=False)

            check = check_angle(rot_fly, angle)

            if check != angle:
                count_flip+=1

            rot_imgs.append(rot_fly)

            #cv.imshow('res', rot_fly)
            #cv.imshow('img',gray)
            #cv.waitKey()
        print('\nWriting video...')

        path = Path(vid_path)
        imgs_dir = os.path.join(path.parent,'crop_imgs')
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        for i, rot_img in enumerate(rot_imgs):
            if count_flip > len(raw_imgs)/2:
                rot_img = ndimage.rotate(rot_img, 180, reshape=False)
                angles[i] = (angles[i]+180)%360
            cv.imwrite(imgs_dir + '/img_' + str(i) + '.jpg', rot_img)

        terminal_call = "ffmpeg -y -loglevel panic -nostats -r "+str(fps)+" -i "+imgs_dir+"/img_%d.jpg -pix_fmt yuv420p -c:v libx265 -x265-params log-level=error -vsync 0 -crf 15 "+ out_path    
        os.system(terminal_call)

        #shutil.rmtree(imgs_dir)
        print('Done!')
    

if __name__ == '__main__':
    main()
