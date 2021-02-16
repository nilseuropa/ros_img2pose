import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from math import cos, sin

from img2pose import img2poseModel
from model_loader import load_model

DEPTH = 18
MIN_SIZE = 100 # 200, 300, 500, 800, 1100, 1400, 1700
MAX_SIZE = 1400

POSE_MEAN = "./models/WIDER_train_pose_mean_v1.npy"
POSE_STDDEV = "./models/WIDER_train_pose_stddev_v1.npy"
MODEL_PATH = "./models/img2pose_v1.pth"

probability_threshold = 0.9
euler_lpf_coeff = 0.33
bb_coord_lpf_coeff = 0.5

np.set_printoptions(suppress=True)
threed_points = np.load('./pose_references/reference_3d_68_points_trans.npy')
transform = transforms.Compose([transforms.ToTensor()])

pose_mean = np.load(POSE_MEAN)
pose_stddev = np.load(POSE_STDDEV)

img2pose_model = img2poseModel(
    DEPTH, MIN_SIZE, MAX_SIZE,
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()

vcap = cv2.VideoCapture(0, cv2.CAP_V4L2)
vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vcap.set(cv2.CAP_PROP_FPS, 30)

def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size = 100):

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

filtered_angles = [0.0,0.0,0.0]
diff_angles = [0.0,0.0,0.0]
filtered_bb_coords = [0.0,0.0,0.0,0.0]
diff_bb_coords = [0.0,0.0,0.0,0.0]

if vcap.isOpened():
    print("Video device opened.")
    is_capturing, frame = vcap.read()

while is_capturing:
    is_capturing, frame = vcap.read()
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
            for n in range(4):
                diff_bb_coords[n] = bbox[n] - filtered_bb_coords[n]
                filtered_bb_coords[n] += diff_bb_coords[n] * bb_coord_lpf_coeff

            bb_x = filtered_bb_coords[0]
            bb_y = filtered_bb_coords[1]
            bb_w = filtered_bb_coords[2] - filtered_bb_coords[0]
            bb_h = filtered_bb_coords[3] - filtered_bb_coords[1]

            pose = res["dofs"].cpu().numpy()[i].astype('float')
            pose = pose.squeeze()
            for n in range(3):
                diff_angles[n] = pose[n] - filtered_angles[n]
                filtered_angles[n] += diff_angles[n] * euler_lpf_coeff

            draw_axis(cv_img, pitch=-filtered_angles[0], yaw=-filtered_angles[1], roll=filtered_angles[2], tdx=bb_x+bb_w/2, tdy=bb_y+bb_h/2, size=100)


    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv2.putText(cv_img, "{:10.4f}".format(1/run_time), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
    cv2.imshow('img2pose',cv_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vcap.release();
        break
