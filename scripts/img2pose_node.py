#!/usr/bin/env python

import rospy
import rospkg
import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from math import cos, sin

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as sensorImage

from ros_img2pose.img2pose import img2poseModel
from ros_img2pose.model_loader import load_model
from ros_img2pose.utils.image_operations import draw_axis

# get absolute paths
rospack = rospkg.RosPack()
models_path = rospack.get_path('ros_img2pose')+"/models/"
POSE_MEAN = models_path+"train_pose_mean_wider.npy"
POSE_STDDEV = models_path+"train_pose_stddev_wider.npy"
MODEL_PATH = models_path+"img2pose_wider.pth"
POINTS_PATH = models_path+"reference_3d_68_points_trans.npy"

probability_threshold = 0.9

np.set_printoptions(suppress=True)
transform = transforms.Compose([transforms.ToTensor()])
threed_points = np.load(POINTS_PATH)
pose_mean = np.load(POSE_MEAN)
pose_stddev = np.load(POSE_STDDEV)

# Load neural model
img2pose_model = img2poseModel(
    18, 100, 1400,
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()

bridge = CvBridge()

def show_image(frame):
    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    start = time.time()
    res = img2pose_model.predict([transform(img)])[0]
    end = time.time()
    run_time = end - start

    for i in range(len(res["scores"])):
        if res["scores"][i] > probability_threshold:
            bbox = res["boxes"].cpu().numpy()[i].astype('int')
            pose = res["dofs"].cpu().numpy()[i].astype('float')
            pose = pose.squeeze()
            bb_x = bbox[0]
            bb_y = bbox[1]
            bb_w = bbox[2] - bbox[0]
            bb_h = bbox[3] - bbox[1]
            draw_axis(cv_img, pitch=-pose[0], yaw=-pose[1], roll=pose[2], tdx=bb_x+bb_w/2, tdy=bb_y+bb_h/2, size=100)

    cv2.putText(cv_img, "{:10.4f}".format(1/run_time), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
    cv2.imshow("img2pose", cv_img)
    cv2.waitKey(1)

def image_callback(img_msg):
    cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    show_image(cv_image)

def estimator():
    rospy.init_node('ros_img2pose', anonymous=True)

    sub_image = rospy.Subscriber("/head_camera/image_raw", sensorImage, image_callback)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__=="__main__":
    try:
        estimator()
    except rospy.ROSInterruptException:
        pass
