import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import matplotlib.pyplot as plt

#============================================
# Camera calibration
#============================================#Define size of chessboard target.
chessboard_size = (9,11)

#Define arrays to save detected points
obj_points_left = [] #3D points in real world space 
img_points_left = [] #3D points in image plane#Prepare grid and points to display
objp_left = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp_left[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

obj_points_right = [] #3D points in real world space 
img_points_right = [] #3D points in image plane#Prepare grid and points to display
objp_right = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp_right[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

#read images
left_camera = glob.glob("C:\\Users\\Zoe\\Desktop\\Images\\checkerboard\\left\\*")
right_camera = glob.glob("C:\\Users\\Zoe\\Desktop\\Images\\checkerboard\\right\\*")

#Iterate over images to find intrinsic matrix
#for image_path in tqdm(calibration_paths):#Load image
print("loading images...")
for image in left_camera:
    #print(image)
    img = cv2.imread(image)
    gray_image_left = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #find chessboard corners
    ret_left,corners_left = cv2.findChessboardCorners(gray_image_left, chessboard_size, None)
    if ret_left == True:
        #define criteria for subpixel accuracy
        criteria_left = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #refine corner location (to subpixel accuracy) based on criteria.
        cv2.cornerSubPix(gray_image_left, corners_left, (5,5), (-1,-1), criteria_left)
        obj_points_left.append(objp_left)
        img_points_left.append(corners_left)
    else:
        print("Chessboard not detected!")
#print(image)

for image in right_camera:
    #print(image)
    img = cv2.imread(image)
    gray_image_right = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #find chessboard corners
    ret_right,corners_right = cv2.findChessboardCorners(gray_image_right, chessboard_size, None)
    if ret_right == True:
        
        #define criteria for subpixel accuracy
        criteria_right = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #refine corner location (to subpixel accuracy) based on criteria.
        cv2.cornerSubPix(gray_image_right, corners_right, (5,5), (-1,-1), criteria_right)
        obj_points_right.append(objp_right)
        img_points_right.append(corners_right)
    else:
        print("Chessboard not detected!")

print("done loading images")
print("calibrating...")
#Calibrate left camera
ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_points_left, img_points_left,gray_image_left.shape[::-1], None, None)

#Calibrate right camera
ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(obj_points_right, img_points_right,gray_image_right.shape[::-1], None, None)

focal_length_left = K_left[0][0]/K_left[1][1]
focal_length_right = K_right[0][0]/K_right[1][1]

#StereoCalibrate

retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(obj_points_left, img_points_left, img_points_right, K_left, dist_left, 
                                                               K_right, dist_right, (1280, 1024), None, None, None, None)
print("done calibrating, beginning rectification")
#rectify both cameras

R1, R2, P1, P2 = cv2.stereoRectify(K1, dist1, K2, dist2, (1280, 1024), R, T, flags=cv2.CALIB_ZERO_DISPARITY)[0:4]

print("undistorting...")
#undistort and compute disparity
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (1280, 1024), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (1280, 1024), cv2.CV_32FC1)
#print(map1x)
print("remapping and saving...")
#for img in range(len(left_camera)):

cv2.imwrite("grayimg.png", gray_image_left)

left_img_rect = cv2.remap(gray_image_left, map1x, map1y, cv2.INTER_LINEAR)
cv2.remap(gray_image_right, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite("test.png", left_img_rect)



#dst = cv2.remap(gray_image, map1x, map1y, cv2.INTER_LINEAR)
#cv2.imwrite("test.png", dst)


