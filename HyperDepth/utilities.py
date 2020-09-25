import os
import matplotlib.pyplot as plt
import pandas as pd
import  scipy.ndimage.filters as filters
import numpy as np



#this file contains utility methods for the HyperDepth implementation


# loads images and extracts their raw feature data
def load_images(file_list, img_path, gt_path, line_idx, displacements, max_radius, n_x):
    # n_x related to random feature selection
    print("loading images + extracting features")
    # preallocate vecs for better speed
    feat_vec_list = np.zeros( [len(line_idx),len(file_list),n_x,len(displacements)], dtype=np.int16 )
    disp_vec_list = np.zeros( [len(line_idx),len(file_list),n_x], dtype=np.int16)
    pixel_vec_list = np.zeros( [len(line_idx),len(file_list),n_x], dtype=np.int16)
    ill_vec_list = np.zeros( [len(line_idx),len(file_list),n_x], dtype=np.float32)

    # loop through all images
    for idx, entry in file_list.iterrows():
        if idx%100 == 0:
            print("Reading image " + idx)
        img_name = "%s/%s.png" % (img_path,entr['hash']);
        disp_name = "%s/%s.png" % (gt_path, entr['hash']);

        image = cv2.imread(img_name, 0)
        disp_map = cv2.imread(disp_name, -1)/16 # cv2 automatically multiplies by 16

        # extract disp features separately
        for l_idx, line in enumerate(line_idx):
            disp_vec_list[l_idx, idx, :] = disp_map[line, max_radius:-max_radius]

        # load in remaining vecs

        temp = extract_image(image, line_idx, displacements, max_radius, n_x)
        feat_vec_list[:, idx, :, :] = temp[0]
        pixel_vec_list[:, idx, :] = temp[1]
        ill_vec_list[:, idx, :] = temp[2]

    return feat_vec_list, disp_vec_list, pixel_vec_list, ill_vec_list

# reads images memory as NP arrays
def read_images(file_list, img_path):
    img_list = []
    
    for idx, entry in file_list.iterrows():
        if idx%100 == 0:
            print ("Reading image " + idx)
        img_name = "%s/%s.png" % (img_path, entry['hash'])

        image = cv2.imread(img_name, 0)
        img_list.append(image)

    return img_list

def list_images(dir):
    print("getting list of images")
    fileList = os.listdir(dir)

    # get list of all valid files in dir
    image_file_list = []
    for file in fileList:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPEG") or file.endswith(".PNG"):
            image_file_list.append(file)

    # generate each file entry and add to list
    entry_list = []
    for file in image_file_list:
        fullName = dir+'/'+file
        fileName = os.path.splitext (file)[0]
        entry_list.append(list([fullName, fileName]))

    file_dframe = pd.DataFrame(list_of_entries, columns = ['filename', 'hash'])
    return file_dframe

# creates a pd.DataFrame from array data
def create_dataframe(feature_vec, displacement_vec, xcoord_vec, intensity_vec):
    print("creating dataframes")
    df_features = pd.DataFrame(np.reshape(feature_vec[-1, feature_vec.shape[2]]))

    df_features['displc'] = np.reshape(displacement_vec, [-1])
    df_features['xcoord'] = np.reshape(xcoord_vec, [-1])
    df_features['intens'] = np.reshape(intensity_vec, [-1])

    df_features = df_features(df_features['displc'] > 0) # filter out undetermined displacements

    # might need more filtering?
    df_features['labels'] = df_features['xcoord'] - df_features['displc']

    return df_features

# calculates TOTAL signal along a line within a given imgslice
# might have to calculate within a sampled window, rather than full slice
def check_signal(imgslice, radius):
    print("checking signal")
    imgslice = np.float32(imgslice)

    # uniform filter pads image edges
    filtered = filters.uniform_filter(imgslice, size=2*radius+1, mode='constant')

    filtered_line = filtered[radius, radius:-radius] * np.square(2*radius+1)
    return filtered_line

def extract_image_feats(image, img_h, img_w, img_dims):
    #init array to return
    #im_feats = np.zeros((img_h, img_w, img_dims), dtype='float32')

    im_feats = [] #not preallocating for now bc of np.append nonsense, fix later for speed
    for line_idx in range(img_h):
        line = image[:][line_idx]
        #line_feats = np.zeros(img_w, dtype='float32')
        line_feats = []
        for pixel in range(img_w):
            #print(line[pixel].shape)
            p = line[pixel]
            line_feats.append(p)
        im_feats.append(line_feats)
    im_feats = np.asarray(im_feats)
    return im_feats


## extracts sets of feature vectors from an image at predefined coords
#def extract_image(image, line_idx, displacements, max_radius, n_x):
#    print("extracting features from image")
#    # preallocating for speed
#    im_feat_vec = np.zeros( [len(line_idx),n_x,len(displacements)], dtype=np.int16 )
#    im_pix_vec = np.zeros( [len(line_idx),n_x], dtype=np.int16)
#    im_ill_vec = np.zeros( [len(line_idx),n_x], dtype=np.float32)
#    x_vec = np.arange( n_x )

#    # did not pad image with 0s -- might need to do later
#    for l_idx, line in enumerate(line_idx):
#        # do I need a ROI if i'm using the whole image?
#        roi = image[line-max_radius:line+max-radius+1:]

#        for disp_idx, dd in enumerate(displacements):
#            im_feat_vec[l_idx, :, disp_idx] = extract_line(roi, [dd[0],dd[1]], [dd[2],dd[3]], max_radius)
#        im_pix_vec[l_idx, :] = x_vec
#        im_ill_vec[l_idx, :] = check_signal(roi, max_radius)

#    return im_feat_vec, im_pix_vec, im_ill_vec

# extracts sets of feature vectors from an NP array of images at predefined coords
def extract_array(images, displacements, max_radius, n_x):
    print("TODO")

# generates pre-defined displacement vectors (? do I need this?)
def generate_displacements(num_features, max_radius):
    print("generating random displacements")
    # might need to handle zero displacement as well
    line_displacements = []

    for disp in range(num_features):
        line_displacements.append(np.random.randint(low = -max_radius, high = max_radius, size = 4)) # size might change
    return line_displacements
    

# calculates pixel difference between displacement vectors along a given line
def extract_line(imgslice, radius, uu, vv):
    # uu, vv = 2D pixel offset values
    print("TODO")

