#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import time
from enum import Enum

from jetson_inference import poseNet, detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawRect, cudaDrawLine, cudaDrawCircle


# Red, Green, Blue, Alpha
ZONE_BOX_COLOR = (157, 225, 157, 100)
ZONE_LINE_WIDTH = 1
ZONE_LINE_COLOR = (157, 225, 157, 255)

HOME_PLATE_MARKER_COLOR_VALID = (0, 255, 0 , 100)
HOME_PLATE_MARKER_COLOR_WARN = (0, 0, 255 , 100)
HOME_PLATE_MARKER_COLOR_EXPIRED = (255, 0, 0 , 100)
HOME_PLATE_MARKER_SIZE = 5

MODELS_PATH = "training/models/homeplate"

# how often, in seconds to run homeplate detection
HOME_PLATE_DETECTION_PERIOD_SEC = 1
HOME_PLATE_AGE_WARN_SEC = 2
HOME_PLATE_AGE_EXPIRED_SEC = 4


class HomePlateDetectAge(Enum):
    VALID = 1
    WARN = 2
    EXPIRED = 3

class HomePlateDetect:

    last_detection_sec = 0
    detection = None
    detect_age = HomePlateDetectAge.EXPIRED

    def __init__(self, detect_net):
        self.detect_net = detect_net  

    def detect(self):
        # Detect Home Plate
        current_time_sec = time.time()
        time_since_last_detection = current_time_sec - self.last_detection_sec
        if time_since_last_detection >= HOME_PLATE_DETECTION_PERIOD_SEC:
            # detect objects in the image (with overlay)
            detections = self.detect_net.Detect(img)

            # print the detections
            #print("detected {:d} objects in image".format(len(detections)))

            for detection in detections:
                if detection.ClassID == 1:
                    print(f"HomePlate Detection: {detection}")
                    self.detection = detection
                    self.last_detection_sec = current_time_sec
                    time_since_last_detection = 0
                    break

        if time_since_last_detection >= HOME_PLATE_AGE_EXPIRED_SEC:
            self.detect_age = HomePlateDetectAge.EXPIRED
        elif time_since_last_detection >= HOME_PLATE_AGE_WARN_SEC:
            self.detect_age = HomePlateDetectAge.WARN
        else:
            self.detect_age = HomePlateDetectAge.VALID

        # print(f"HomePlate Detect Age: {self.detect_age}")

    def draw_markers(self,img):
        if self.detect_age == HomePlateDetectAge.WARN:
            color = HOME_PLATE_MARKER_COLOR_WARN
        elif self.detect_age == HomePlateDetectAge.EXPIRED:
            color = HOME_PLATE_MARKER_COLOR_EXPIRED
        else:
            color = HOME_PLATE_MARKER_COLOR_VALID

        # only draw markers if we have detection points previously
        if self.detection:
            cudaDrawCircle(img, (self.detection.Left, self.detection.Bottom), HOME_PLATE_MARKER_SIZE, color)
            cudaDrawCircle(img, (self.detection.Right, self.detection.Bottom), HOME_PLATE_MARKER_SIZE, color)

    def expired(self):
        return (self.detect_age == HomePlateDetectAge.EXPIRED)

def left_and_right_keypoint(pose, left_keypoint, right_keypoint):
    # return [(x,y)] of coordinates for keypoints with left and right side.
    # Note: List may be of length 0, 1 or 2.

    xy = []

    left_idx = pose.FindKeypoint(left_keypoint)
    right_idx = pose.FindKeypoint(right_keypoint)

    if left_idx > 0:
        xy.append((pose.Keypoints[left_idx].x, pose.Keypoints[left_idx].y))

    if right_idx > 0:
        xy.append((pose.Keypoints[right_idx].x, pose.Keypoints[right_idx].y))

    return xy


def is_in_stance(pose):
    elbows = left_and_right_keypoint(pose, 'left_elbow', 'right_elbow')
    wrists = left_and_right_keypoint(pose, 'left_wrist', 'right_wrist')

    # Need to have at least 1 elbow and wrist, otherwise not in stance
    if len(elbows) == 0 or len(wrists) == 0:
        print(f"len elbows: {len(elbows)}. len wrists: {len(wrists)}")
        return False

    # unpack into list of y coordinates
    _, wrists_y = zip(*wrists)
    _, elbows_y = zip(*elbows)

    print(f"wrists_y: {wrists_y}.elbows_y: {elbows_y}")
    # use lowest wrist and highest elbow
    if max(wrists_y) < min(elbows_y):
        return True
    else:
        return False


def get_zone(pose, homeplate_detection):
    # return (left, top, right, bottom) tuple strike zone box

    # top of zone is midpoint between shoulder_y and hip_y
    # bottom of zone is knees_y
    # left zone is left of home plate
    # right zone is right of home plate

    shoulders = left_and_right_keypoint(
        pose, 'left_shoulder', 'right_shoulder')
    hips = left_and_right_keypoint(pose, 'left_hip', 'right_hip')
    knees = left_and_right_keypoint(pose, 'left_knee', 'right_knee')

    # Need to have at least 1 shoulder, hip and knee, otherwise
    if len(shoulders) == 0 or len(hips) == 0 or len(knees) == 0:
        return None

    # unpack into list of y coordinates
    _, shoulders_y = zip(*shoulders)
    _, hips_y = zip(*hips)
    knees_x, knees_y = zip(*knees)

    # we will create a taller zone by choosing the top most shoulder and bottom most hip
    shoulder_y = min(shoulders_y)
    hips_y = max(hips_y)
    top_zone = ((hips_y - shoulder_y)) / 2 + shoulder_y

    # use lowest knee for bottom of zone
    bottom_zone = max(knees_y)

    right_zone = homeplate_detection.Right
    left_zone = homeplate_detection.Left

    return (left_zone, top_zone, right_zone, bottom_zone)

def draw_zone(img, zone): 
    (left, top, right, bottom) = zone

    cudaDrawRect(img, zone, ZONE_BOX_COLOR)

    cudaDrawLine(img, (left,top), (right,top), ZONE_LINE_COLOR, ZONE_LINE_WIDTH)
    cudaDrawLine(img, (left,bottom), (right,bottom), ZONE_LINE_COLOR, ZONE_LINE_WIDTH)
    cudaDrawLine(img, (left,top), (left,bottom), ZONE_LINE_COLOR, ZONE_LINE_WIDTH)
    cudaDrawLine(img, (right,top), (right,bottom), ZONE_LINE_COLOR, ZONE_LINE_WIDTH)


# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="",
                    nargs='?', help="URI of the output stream")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the pose estimation model
pose_net = poseNet(network="resnet18-body", threshold=0.15)
detect_net =  detectNet(model=f"{MODELS_PATH}/ssd-mobilenet.onnx", labels=f"{MODELS_PATH}/labels.txt", 
                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                 threshold=0.3)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

print(f">>> sys.argv: {sys.argv}")
print(f">>> input: {args.input}")
print(f">>> output: {args.output}")

home_plate = HomePlateDetect(detect_net)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None:  # timeout
        continue

    # do the home plate detection
    home_plate.detect()

    # Detect and Draw strike zone if home_plate is detected
    if not home_plate.expired():
        # perform pose estimation (with overlay)
        poses = pose_net.Process(img, overlay="links,keypoints")

        # print the pose results
        # print("PoseNet detected {:d} objects in image".format(len(poses)))

        for pose in poses:
            # print(pose)
            # print(pose.Keypoints)
            # print('Links', pose.Links)

            if (is_in_stance(pose)):
                zone = get_zone(pose, home_plate.detection)
                if zone:
                    draw_zone(img, zone)
                else:
                    print('No Strike Zone')
            else:
                print('Not in stance!')
            
    home_plate.draw_markers(img)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("PoseNet Network {:.0f} FPS".format(
        pose_net.GetNetworkFPS()))

    # print out performance info
    #pose_net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
