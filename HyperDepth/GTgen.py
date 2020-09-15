import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import utilities as utils
import PatchMatch as pm

data_path = "C:\\Users\\Zoe\\Documents\\Thesis\\Test Data\\playground-hyperdepth-master-images"
save_path = "C:\\Users\\Zoe\\Documents\\Thesis\\Test Data\\playground-hyperdepth-GT"
ref_path = "" #not sure what ref image will be yet

max_displacement = 128
block_size = 21

# make sure save path exists
try:
    os.mkdir( save_path )
except:
    pass

# read in ref image
img_ref = cv2.imread(ref_path, 0)

# get list of images
img_list = utils.list_images(data_path)



# loop over all images & generate disparity map

for idx, entry in img_list.iterrows():
    print("idx = " + idx)
    img = cv2.imread(entry['filename'], 0)
    pm_result = pm.main(img, img_ref)

    #calc disparity from pm_result?


    # save disparity map to disk
    disp_map[disp_map<0] = 0 # remove any <0 values
    file_name = "%s/%s.png" % (save_path,entry['hash'])
    cv2.imread(file_name, disp_map.astype(np.uint16)) # 16 bit for now, might change

    # plot disparity maps every 25 images
    if idx % 25 == 0:
        plt.figure(1)
        plt.imshow(img)
        plt.show(block = False)
        plt.pause(0.5)

        plt.figure(2)
        plt.imshow(disp_map/16.0)
        plt.clim(-16,256/2)
        plt.show(block = False )
        plt.pause(0.5)
