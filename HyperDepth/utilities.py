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
#def create_dataframe(feature_vec, displacement_vec, xcoord_vec, intensity_vec):
#    print("creating dataframes")
#    df_features = pd.DataFrame(np.reshape(feature_vec[-1, feature_vec.shape[2]]))

#    df_features['displc'] = np.reshape(displacement_vec, [-1])
#    df_features['xcoord'] = np.reshape(xcoord_vec, [-1])
#    df_features['intens'] = np.reshape(intensity_vec, [-1])

#    df_features = df_features(df_features['displc'] > 0) # filter out undetermined displacements

#    # might need more filtering?
#    df_features['labels'] = df_features['xcoord'] - df_features['displc']

#    return df_features

# calculates TOTAL signal along a line within a given imgslice
# might have to calculate within a sampled window, rather than full slice
def check_signal(imgslice, radius):
    print("checking signal")
    imgslice = np.float32(imgslice)

    # uniform filter pads image edges
    filtered = filters.uniform_filter(imgslice, size=2*radius+1, mode='constant')

    filtered_line = filtered[radius, radius:-radius] * np.square(2*radius+1)
    return filtered_line

#extracts features from a single image
def extract_image_feats(image, img_h, img_w, img_dims):
    #init array to return
    #im_feats = np.zeros((img_h, img_w, img_dims), dtype='float32')

    im_feats = [] #not preallocating for now bc of np.append nonsense, fix later for speed

    for line_idx in range(img_h):

        if img_dims == 1:
            #print("single channel")
            line = image[0][:][line_idx]
            #print(line.shape)
        else:
           # print("color img")
            line = image[:][line_idx]
            #print(line.shape)

        #line_feats = np.zeros(img_w, dtype='float32')
        line_feats = []
        for pixel in range(img_w):
            #print(line[pixel].shape)
            p = line[pixel]
            line_feats.append(p)
        im_feats.append(line_feats)
    im_feats = np.asarray(im_feats)
    return im_feats

#extracts a single line of features from an image at given index
def extract_line(image, img_w, img_dims, idx):

    feats = []

    if img_dims == 1:
        #print("single channel")
        line = image[0][:][idx]
        #print(line.shape)
    else:
        # print("color img")
        line = image[:][idx]
        #print(line.shape)

    #line_feats = np.zeros(img_w, dtype='float32')
    for pixel in range(img_w):
        #print(line[pixel].shape)
        p = line[pixel]
        feats.append(p)
    return feats

#def eval_accuracy(train, test, model, verbose=True):





