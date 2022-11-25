import cv2
import collections
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
import mss
import numpy as np

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

angles_avg_store = collections.deque(maxlen=2)

# measured calibration parameters found from printing angles variable (tilt, pan, roll)
pan_right = -30
pan_left = 15
tilt_neutral = -5
tilt_down = -15
tilt_up = 8

# virtual screen parameters (found from sct.monitors)
left_mon = 2
center_mon = 3
right_mon = 4
screen_width = 1920
screen_height = 1080

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

normed_pan_angle = 0
normed_pitch_angle = 0

# translate from one range to another
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

            # calculate face angles
            rmat, jac = cv2.Rodrigues(pose[0])
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            angles_avg_store.append(angles)

            # find the moving average angles to reduce shake
            averaged_pan_angle = sum(el[1] for el in angles_avg_store) / len(angles_avg_store)
            averaged_pitch_angle = sum(el[0] for el in angles_avg_store) / len(angles_avg_store)

            # Convert raw angles to range [-1,1] where 1 is right screen full, -1 is left screen full
            normed_pan_angle = translate(averaged_pan_angle, pan_right, pan_left, -1, 1)
            normed_pan_angle = -normed_pan_angle
            if normed_pan_angle > 1:
                normed_pan_angle = 1
            if normed_pan_angle < -1:
                normed_pan_angle = -1

            # normalise and clip pitch angles using piecewise linear functions
            if averaged_pitch_angle < tilt_neutral:
                normed_pitch_angle = (averaged_pitch_angle - tilt_neutral) / (tilt_neutral - tilt_down)
            else:
                normed_pitch_angle = (averaged_pitch_angle - tilt_neutral) / (tilt_up - tilt_neutral)

            if normed_pitch_angle > 1:
                normed_pitch_angle = 1
            if normed_pitch_angle < -1:
                normed_pitch_angle = -1

        # looking left case
        if normed_pan_angle < 0:
            img_left_mon = np.array(sct.grab(sct.monitors[left_mon]))
            img_center_mon = np.array(sct.grab(sct.monitors[center_mon]))
            sliced_img_left_mon = img_left_mon[:, int((1 - abs(normed_pan_angle)) * screen_width):, :]
            sliced_img_center_mon = img_center_mon[:, 0:int((1 - abs(normed_pan_angle)) * screen_width), :]

            img = np.concatenate((sliced_img_left_mon, sliced_img_center_mon), axis=1)

        # looking right case
        if normed_pan_angle >= 0:
            img_right_mon = np.array(sct.grab(sct.monitors[right_mon]))
            img_center_mon = np.array(sct.grab(sct.monitors[center_mon]))
            sliced_img_right_mon = img_right_mon[:, 0:int(abs(normed_pan_angle) * screen_width), :]
            sliced_img_center_mon = img_center_mon[:, int(abs(normed_pan_angle) * screen_width):, :]

            img = np.concatenate((sliced_img_center_mon, sliced_img_right_mon), axis=1)

        # looking up case
        if normed_pitch_angle > 0:
            # shift image down, replacing by black
            showing_screen = img[0:int((1 - abs(normed_pitch_angle)) * screen_height), :, :]
            showing_black = np.zeros(
                (screen_height - int((1 - abs(normed_pitch_angle)) * screen_height), screen_width, 4), dtype=np.uint8)
            showing_black[:, :, 3] = 255
            img = np.concatenate((showing_black, showing_screen), axis=0)

        # looking down case
        if normed_pitch_angle <= 0:
            # shift image up, replacing by black
            showing_screen = img[int(abs(normed_pitch_angle) * screen_height):, :, :]
            showing_black = np.zeros((int(abs(normed_pitch_angle) * screen_height), screen_width, 4), dtype=np.uint8)
            showing_black[:, :, 3] = 255
            img = np.concatenate((showing_screen, showing_black), axis=0)

        # Display the picture
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow("window", -500, 0) # this could be used to automatically move window to a new screen
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
