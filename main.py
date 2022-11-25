"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
import mss
import numpy as np
print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))




# Before estimation started, there are some startup works to do.

# 1. Setup the video source from webcam

video_src = 1

cap = cv2.VideoCapture(video_src)

# Get the frame size. This will be used by the pose estimator.
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 2. Introduce a pose estimator to solve pose.
pose_estimator = PoseEstimator(img_size=(height, width))

# 3. Introduce a mark detector to detect landmarks.
mark_detector = MarkDetector()

# 4. Measure the performance with a tick meter.
tm = cv2.TickMeter()

# Now, let the frames flow.
mon = 3

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

with mss.mss() as sct:

    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))

            # Do you want to see the head axes?
            # pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

            rmat, jac = cv2.Rodrigues(pose[0])
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Convert raw angles to range [-1,1] where 1 is right screen full, -1 is left screen full
            normed_angle = translate(angles[1], -30, 15, -1, 1)
            normed_angle = -normed_angle
            if normed_angle>1:
                normed_angle = 1
            if normed_angle < -1:
                normed_angle = -1


        left_mon = 2
        center_mon = 3
        right_mon = 4
        screen_width = 1920
        print(normed_angle)
        if normed_angle < 0:
            img_left_mon = np.array(sct.grab(sct.monitors[left_mon]))
            img_center_mon = np.array(sct.grab(sct.monitors[center_mon]))
            sliced_img_left_mon = img_left_mon[:, int((1 - abs(normed_angle)) * screen_width):, :]
            sliced_img_center_mon = img_center_mon[:, 0:int((1 - abs(normed_angle)) * screen_width), :]

            img = np.concatenate((sliced_img_left_mon, sliced_img_center_mon), axis=1)

        if normed_angle >= 0:
            img_right_mon = np.array(sct.grab(sct.monitors[left_mon]))
            img_center_mon = np.array(sct.grab(sct.monitors[center_mon]))
            sliced_img_right_mon = img_right_mon[:, 0:int(abs(normed_angle) * screen_width), :]
            sliced_img_center_mon = img_center_mon[:, int(abs(normed_angle) * screen_width):, :]

            img = np.concatenate((sliced_img_center_mon, sliced_img_right_mon), axis=1)


        # Display the picture
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # Show preview.
        # cv2.imshow("Preview", frame)
        # if cv2.waitKey(1) == 27:
        #     break
