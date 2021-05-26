import io
import time
import picamera
import sys
import os

class SplitFrames(object):
    def __init__(self,folder):
        self.frame_num = 0
        self.output = None
        self.folder = folder

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open(self.folder+'/img_%03d.jpg' % self.frame_num, 'wb')
        self.output.write(buf)

folder = sys.argv[-1]

if not os.path.isdir(folder):
    os.mkdir(folder)
else:
    print('Folder already exists!')
    os.quit()

experiment_time = 30 #s

with picamera.PiCamera() as camera:
    camera.resolution = (800,800)
    camera.framerate = 81
    camera.shutter_speed = 2000
    camera.start_preview()
    # Give the camera some warm-up time
    time.sleep(2)
    output = SplitFrames(folder)
    print('start...')
    start = time.time()
    camera.start_recording(output, format='mjpeg')
    camera.wait_recording(experiment_time)
    camera.stop_recording()
    finish = time.time()
print('Captured %d frames at %.2ffps' % (
    output.frame_num,
    output.frame_num / (finish - start)))
