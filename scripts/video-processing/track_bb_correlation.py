#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example shows how to use the correlation_tracker from the dlib Python
# library.  This object lets you track the position of an object as it moves
# from frame to frame in a video sequence.  To use it, you give the
# correlation_tracker the bounding box of the object you want to track in the
# current video frame.  Then it will identify the location of the object in
# subsequent frames.
#
# In this particular example, we are going to run on the
# video sequence that comes with dlib, which can be found in the
# examples/video_frames folder.  This video shows a juice box sitting on a table
# and someone is waving the camera around.  The task is to track the position of
# the juice box as the camera moves around.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import os
import glob
from imutils.video import FPS
import dlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

mid_folder = Path('/Volumes/MyWorld/FIW-MM/data/interm/visual/video-frame-faces/F0005/MID3')
# Path to the video frames
video_folder = mid_folder.joinpath('76dkmAvXHPE_0000001')
vidoe_file = Path('/Volumes/MyWorld/FIW-MM/data/FIDs-MM/visual/video/F0005/MID3/76dkmAvXHPE_0000001.mp4')

f_faces = list(video_folder.glob('*.jpg'))
f_faces.sort()
# Create the correlation tracker - the object needs to be initialized
# before it can be used
tracker = dlib.correlation_tracker()
# frame-000-00-01-bb.csv
# frame-000-00-01-landmarks.csv
# frame-000-00-01.jpg
# frame-000-00-bb.csv
# frame-000-00-landmarks.csv
# frame-000-00.jpg

# win = dlib. image_window()
# We will track the frames as we load them off of disk
# screen_res = 1280, 720
# initialize the video stream, dlib correlation tracker, output video
# writer, and predicted class label

# initialize the video stream, dlib correlation tracker, output video
# writer, and predicted class label
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(str(vidoe_file))
writer = None
label = ""
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
f = f_faces[0]
k = 0
while True:
    # grab the next frame from the video file
    (grabbed, frame) = vs.read()
    # check to see if we have reached the end of the video file
    if frame is None:
        break
    # resize the frame for faster processing and then convert the
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if we are supposed to be writing a video to disk, initialize
    # the writer

    # for k, f in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
    print("Processing Frame {}".format(k))
    # img = dlib.load_rgb_image(str(f))

    # We need to initialize the tracker on the first frame
    if k == 0:
        # Start a track on the juice box. If you look at the first frame you
        # will see that the juice box is contained within the bounding
        # box (74, 67, 112, 153).
        bb = np.loadtxt(str(f).replace('.jpg', '-bb.csv'), delimiter=',')
        bb_in = tuple([int(coord) for coord in bb[:-1]])
        rect = dlib.rectangle(*bb_in)
        tracker.start_track(rgb, rect)

        # tracker.start_track(img, dlib.rectangle(74, 67, 112, 153))153
    else:
        # Else we just attempt to track from the previous frame
        tracker.update(rgb)

        bb_new = tracker.get_position()

    pil_frame = Image.fromarray(rgb)
    bl = bb_new.bl_corner()
    tr = bb_new.tr_corner()
    pil_frame.crop((bl.x, tr.y, tr.x, bl.y))
    pil_frame.crop((int(bb_new.left), int(bb_new.top), int(bb_new.right), int(bb_new.bottom)))
    # if args["output"] is not None and writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter(args["output"], fourcc, 30,
    #                              (frame.shape[1], frame.shape[0]), True)
    #
    # scale_width = screen_res[0] / img.shape[1]
    # scale_height = screen_res[1] / img.shape[0]
    # scale = min(scale_width, scale_height)
    # window_width = int(img.shape[1] * scale)
    # window_height = int(img.shape[0] * scale)
    # cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst_rt', window_width, window_height)
    # cv2.imshow('dst_rt', rgb)
    # cv2.waitKey(0)
    k += 1
    print(tracker)
    # cv2.destroyAllWindows()
    # # win.clear_overlay()
    # # win.set_image(img)
    # # win.add_overlay(tracker.get_position())
    # dlib.hit_enter_to_continue()
