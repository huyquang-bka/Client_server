import os
import cv2
import numpy as np
import math
from skimage import transform

# import the necessary packages
from imutils.video import VideoStream
import time
import imutils

# Load Model
# with tf.device('/GPU:0'):


print("Loaded")

# C9_day PKSpace coordinates (other images coordinates could be offset by a few pixels)
# First row 130x130
# Should write each x-y pair on the same line instead of multiple lines
# Initialize, skip arr[0] for counting convenience in while loop
# x_min = [];
# y_min = [];
# x_max = [];
# y_max = []
# x_min.append(0);
# y_min.append(0);
# x_max.append(0);
# y_max.append(0)  # Redundant
# # First row 130x130
# x_min.append(18);
# y_min.append(732)  # 1
# x_max.append(x_min[1] + 130);
# y_max.append(y_min[1] + 130)
# x_min.append(216);
# y_min.append(750)  # 2
# x_max.append(x_min[2] + 130);
# y_max.append(y_min[2] + 130)
# x_min.append(346);
# y_min.append(750)  # 3
# x_max.append(x_min[3] + 130);
# y_max.append(y_min[3] + 130)
# x_min.append(490);
# y_min.append(756)  # 4
# x_max.append(x_min[4] + 130);
# y_max.append(y_min[4] + 130)
# x_min.append(636);
# y_min.append(768)  # 5
# x_max.append(x_min[5] + 130);
# y_max.append(y_min[5] + 130)
# x_min.append(784);
# y_min.append(762)  # 6
# x_max.append(x_min[6] + 130);
# y_max.append(y_min[6] + 130)
# x_min.append(934);
# y_min.append(762)  # 7
# x_max.append(x_min[7] + 130);
# y_max.append(y_min[7] + 130)
# x_min.append(1234);
# y_min.append(762)  # 8
# x_max.append(x_min[8] + 130);
# y_max.append(y_min[8] + 130)
# x_min.append(1372);
# y_min.append(742)  # 9
# x_max.append(x_min[9] + 130);
# y_max.append(y_min[9] + 130)
# x_min.append(1584);
# y_min.append(718)  # 10
# x_max.append(x_min[10] + 130);
# y_max.append(y_min[10] + 130)
# x_min.append(1684);
# y_min.append(700)  # 11
# x_max.append(x_min[11] + 130);
# y_max.append(y_min[11] + 130)
# # Second row 90x90
# x_min.append(432);
# y_min.append(512)  # 12
# x_max.append(x_min[12] + 90);
# y_max.append(y_min[12] + 90)
# x_min.append(528);
# y_min.append(512)  # 13
# x_max.append(x_min[13] + 90);
# y_max.append(y_min[13] + 90)
# x_min.append(624);
# y_min.append(512)  # 14
# x_max.append(x_min[14] + 90);
# y_max.append(y_min[14] + 90)
# x_min.append(724);
# y_min.append(512)  # 15
# x_max.append(x_min[15] + 90);
# y_max.append(y_min[15] + 90)
# x_min.append(824);
# y_min.append(506)  # 16
# x_max.append(x_min[16] + 90);
# y_max.append(y_min[16] + 90)
# x_min.append(923);
# y_min.append(508)  # 17
# x_max.append(x_min[17] + 90);
# y_max.append(y_min[17] + 90)
# # Third row 75x75
# x_min.append(417);
# y_min.append(455)  # 18
# x_max.append(x_min[18] + 75);
# y_max.append(y_min[18] + 75)
# x_min.append(502);
# y_min.append(455)  # 19
# x_max.append(x_min[19] + 75);
# y_max.append(y_min[19] + 75)
# x_min.append(590);
# y_min.append(455)  # 20
# x_max.append(x_min[20] + 75);
# y_max.append(y_min[20] + 75)
# x_min.append(680);
# y_min.append(453)  # 21
# x_max.append(x_min[21] + 75);
# y_max.append(y_min[21] + 75)
# x_min.append(765);
# y_min.append(453)  # 22
# x_max.append(x_min[22] + 75);
# y_max.append(y_min[22] + 75)
# x_min.append(853);
# y_min.append(453)  # 23
# x_max.append(x_min[23] + 75);
# y_max.append(y_min[23] + 75)
# x_min.append(946);
# y_min.append(451)
# x_max.append(x_min[24] + 75);
# y_max.append(y_min[24] + 75)
# # 24
# # Sensor
# x_min.append(164);
# y_min.append(509)
# x_min.append(1304);
# y_min.append(765)
#
# x_max.append(x_min[25] + 90);
# y_max.append(y_min[25] + 90)
# x_max.append(x_min[26] + 90);
# y_max.append(y_min[26] + 90)
#
# cap_time = []
# cut_time = []
# pre_time = []

# cap = cv2.VideoCapture("rtsp://admin:bk123456@192.168.0.90:554/Streaming/channels/1/")

# cap.open("rtsp://admin:bk123456@192.168.0.90:554/Streaming/channels/1/")
# with tf.device('/device:GPU:0'):
vs = VideoStream("rtsp://admin:bk123456@192.168.0.155:554/Streaming/channels/1/").start()
a = 0

while (a < 101):
    # with tf.device('/GPU:0'):
    t_cut = 0
    t_pre = 0
    count_busy = 0
    count_free = 0

    start = time.time()
    # cap = cv2.VideoCapture("rtsp://admin:bk123456@192.168.0.90:554/Streaming/channels/1/")
    frame = vs.read()
    # ret, frame = cap.read()

    # frame = cv2.imread('/home/jn/JN/datn1/C9_in.jpg' )

    # for i in range(1, 25):
    #     #    with tf.device('/device:CPU:0'):
    #
    #     cv2.rectangle(frame, (x_min[i], y_min[i]), (x_max[i], y_max[i]), (0, 255, 0), 3)
    #
    #     # cut.append(end_cut - start_cut)
    #
    #     # pre.append(end_pre - end_cut)
    #
    # frame = imutils.resize(frame, width=int(720 * 1.5), height=int(480 * 1.5))
    out =
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print('Total busy slot: %d\n' % count_busy)
    # print('Total free slot: %d\n' % count_free)

    # t = str(end - start)

# cap.release()
cv2.destroyAllWindows()


