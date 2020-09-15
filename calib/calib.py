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
circles_grid_size = (5,4)

#Define arrays to save detected points
obj_points_left = [] #3D points in real world space 
img_points_left = [] #2D points in image plane

proj_obj_points = [] #projector 3D points in real world space
proj_img_points = [] #projector 2D points in image plane

# temp array for 3D points -- camera
objp_left = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp_left[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# temp array for 3D points -- projector
proj_objp = np.zeros((np.prod(circles_grid_size),3),dtype=np.float32)
proj_objp[:,:2] = np.mgrid[0:circles_grid_size[0], 0:circles_grid_size[1]].T.reshape(-1,2)

#define criteria for subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#read images
left_camera = glob.glob("C:\\Users\\Zoe\\Desktop\\calibimgs\\test_small\\*")


# downsample images to projector resolution??

def intersectCircRayToBoard(circles, rvec, tvec, K, dist):
    circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist))
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    if not rvec.size:
        return None

    R, _ = cv2.Rodrigues(rvec)
 
#stackoverflow.com/questions/5666222/3d-line-plane-intersection
 
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane

    plane_point = tvec.T     # t is a point on the plane
    plane_point = np.reshape(plane_point, (1, 3, 1))
    epsilon = 1e-06
 
    circles_3d = np.zeros((0,3), dtype=np.float32)
 
    for p in circles_normalized:
        ray_direction = p / np.linalg.norm(p)
        ray_point = p
 
        ndotu = plane_normal.dot(ray_direction.T)
 
        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
 
        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        Psi = w + si * ray_direction + plane_point
        Psi = np.reshape(Psi, (3, 3))
        circles_3d = np.append(circles_3d, Psi, axis = 0).astype('float32')
    return circles_3d

#Iterate over images to find intrinsic matrix
print("loading images...")

print("finding corners...")
for image in left_camera:
    img_left = cv2.imread(image)
    gray_image_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    #find chessboard corners
    ret_left,corners_left = cv2.findChessboardCorners(gray_image_left, chessboard_size, None)


    if ret_left == True:
        obj_points_left.append(objp_left)

        #refine corner location (to subpixel accuracy) based on criteria.
        corners2 = cv2.cornerSubPix(gray_image_left, corners_left, (5,5), (-1,-1), criteria)
        
        img_points_left.append(corners2)
    else:
        print("Chessboard not detected!")


print("calibrating camera...")
ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_points_left, img_points_left,
                                                                          gray_image_left.shape[::-1], None, None)
rvecs_array = np.asarray(rvecs_left)
tvecs_array = np.asarray(tvecs_left)
print("finding circles...")

img_counter = 0
for image in left_camera:
    proj_img = cv2.imread(image)
    gray_proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2GRAY)

    #find circles
    ret_circ, circles = cv2.findCirclesGrid(gray_proj_img, circles_grid_size, cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret_circ == True:
        proj_obj_points.append(proj_objp)

        corners2 = cv2.cornerSubPix(gray_proj_img, circles, (5,5), (-1,-1), criteria)

        circles3D = intersectCircRayToBoard(circles, rvecs_array[img_counter], tvecs_array[img_counter], K_left, dist_left)
        proj_img_points.append(corners2)
    
    else:
        print("no circles detected!")
    img_counter+= 1

# calibrate projector
print("calibrating projector...")

ret_proj, K_proj, dist_proj, rvecs_proj, tvecs_proj = cv2.calibrateCamera(proj_obj_points, proj_img_points, (1024, 768), None, None)

print("stereo calibrating...")
# stereo calibration
retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(obj_points_left, img_points_left, proj_img_points, K_left, 
                                                               dist_left, K_proj, dist_proj, (1280, 1024), 
                                                               None, None, None, None, cv2.CALIB_USE_INTRINSIC_GUESS)

print("done calibrating, beginning rectification...")
# stereo rectification
R1, R2, P1, P2, Q = cv2.stereoRectify(K1, dist1, K2, dist2, (1280, 1024), R, T, None, None, None, None, None, 
                                      0, -1, (1280, 1024))[0:5]

print("undistorting...")
# undistort left & right
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (1280, 1024), cv2.CV_32FC1)


print("remapping...")
# remap final results
left_img_rect = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
right_img_rect = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

fig, ax = plt.subplots(nrows=2, ncols=2)

plt.subplot(2, 2, 1)
plt.imshow(img_left)

plt.subplot(2, 2, 2)
plt.imshow(img)

plt.subplot(2, 2, 3)
plt.imshow(left_img_rect)

plt.subplot(2, 2, 4)
plt.imshow(right_img_rect)

plt.show(block=False)
plt.pause(5)
plt.close()


#cv2.imwrite("grayimg.png", gray_image_left)
#cv2.imwrite("test.png", left_img_rect)



#dst = cv2.remap(gray_image, map1x, map1y, cv2.INTER_LINEAR)
#cv2.imwrite("test.png", dst)


