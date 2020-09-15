import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[0:2]
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        img1 = cv2.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2

def main():
    debug = False
    cleaned_file = []
    K1 = []
    K2 = []
    dist1 = []
    dist2 = []
    R = []
    T = []
    calib_file = [np.loadtxt("C:\\Users\\Zoe\\Desktop\\brown_calibration\\cropped_sets\\cropped_results2.txt", dtype="str", delimiter = "\n")]
    img_left = cv2.imread("C:\\Users\\Zoe\\Desktop\\brown_calibration\\checkerboards\\pattern_rotated.png")
    img_right = cv2.imread("C:\\Users\\Zoe\\Desktop\\brown_calibration\\checkerboards\\checkerboard_wall.jpg")


    for item in calib_file:
        for i in range(len(item)):
            cleaned_file.append(item[i].split(", "))

    for item in cleaned_file:
        for i in range(len(item)):
            item[i] = float(item[i])

    K1 = np.array(cleaned_file[0:3])

    dist1 = np.array(cleaned_file[3])
    K2 = np.array(cleaned_file[4:7])
    dist2 = np.array(cleaned_file[7])
    R = np.array(cleaned_file[8:11])
    T = np.array(cleaned_file[11])

    focal_length_left = K1[0][0]/K1[1][1]
    focal_length_right = K2[0][0]/K2[1][1]
    
    # stereo rectification

    R1, R2, P1, P2, Q = cv2.stereoRectify(K1, dist1, K2, dist2, (1024, 1280), R, T, flags=cv2.CALIB_ZERO_DISPARITY, 
                                          alpha=0)[0:5]
    if debug:
        print("R1 = ", R1)
        print("R2 = ", R2)
        print("P1 = ", P1)
        print("P2 = ", P2)
        print("FL left = ", focal_length_left)
        print("FL right = ", focal_length_right)
        print(len(dist1))
        print("dist1 = ", dist1)
        print("dist2 = ", dist2)
        print("Q = ", Q)

    # undistort and remap
    #map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (1024, 1280), cv2.CV_32FC1)
    #map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (1024, 1280), cv2.CV_32FC1)

    undist_1 = cv2.undistort(img_left, K1, dist1)
    undist_2 = cv2.undistort(img_right, K2, dist2)
    

    fig, ax = plt.subplots(nrows=2, ncols=2)

    fig.set_size_inches(10,10)

    plt.subplot(2, 2, 1)
    plt.imshow(img_left)

    plt.subplot(2, 2, 2)
    plt.imshow(img_right)

    plt.subplot(2, 2, 3)
    #plt.imshow(left_img_rect)
    plt.imshow(undist_1)

    plt.subplot(2, 2, 4)
    #plt.imshow(right_img_rect)
    plt.imshow(undist_2)

    plt.show(block=False)
    plt.pause(7)
    plt.close()



   # #calculate key points & compute epilines
   # sift = cv2.xfeatures2d.SIFT_create()

   # # find the keypoints and descriptors with SIFT

   # kp1, des1 = sift.detectAndCompute(img_left,None)
   # kp2, des2 = sift.detectAndCompute(img_right,None)

   # # FLANN parameters
   # FLANN_INDEX_KDTREE = 1
   # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   # search_params = dict(checks=50)
   # flann = cv2.FlannBasedMatcher(index_params,search_params)
   # matches = flann.knnMatch(des1,des2,k=2)
   # good = []
   # pts1 = []
   # pts2 = []

   # # ratio test as per Lowe's paper
   # for i,(m,n) in enumerate(matches):
   #     if m.distance < 0.96*n.distance:
   #         good.append(m)
   #         pts2.append(kp2[m.trainIdx].pt)
   #         pts1.append(kp1[m.queryIdx].pt)


   # pts1 = np.int32(pts1)
   # pts2 = np.int32(pts2)

   # F, mask = cv2.findFundamentalMat(pts1,pts2)
   # print(F)
   # F = F.T
   # print(F)
   # # We select only inlier points
   # pts1 = pts1[mask.ravel()==1]
   # pts2 = pts2[mask.ravel()==1]


   # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
   # lines1 = lines1.reshape(-1,3)
   # img5,img6 = drawlines(img_left,img_right,lines1,pts1,pts2)

   # # Find epilines corresponding to points in left image (first image) and
   # # drawing its lines on right image
   # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
   # lines2 = lines2.reshape(-1,3)
   # img3,img4 = drawlines(img_right,img_left,lines2,pts2,pts1)

   # lines3 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
   # lines3 = lines3.reshape(-1, 3)
   # img7, img8 = drawlines(undist_1, undist_2, lines3, pts1, pts2)

   # lines4 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
   # lines4 = lines4.reshape(-1, 3)
   # img9, img10 = drawlines(undist_2, undist_1, lines4, pts1, pts2)


   ## plt.subplot(121),plt.imshow(img7)
   # #plt.subplot(122),plt.imshow(img9)
   # #plt.show()


   # #left_img_rect = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
   # #right_img_rect = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)


   # fig, ax = plt.subplots(nrows=2, ncols=2)

   # fig.set_size_inches(10,10)

   # plt.subplot(2, 2, 1)
   # plt.imshow(img_left)

   # plt.subplot(2, 2, 2)
   # plt.imshow(img_right)

   # plt.subplot(2, 2, 3)
   # #plt.imshow(left_img_rect)
   # plt.imshow(undist_1)

   # plt.subplot(2, 2, 4)
   # #plt.imshow(right_img_rect)
   # plt.imshow(undist_2)

   # plt.show(block=False)
   # plt.pause(7)
   # plt.close()

main()