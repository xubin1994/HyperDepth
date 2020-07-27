import os
import matplotlib.pyplot as plt
import pandas as pd



#this file contains utility methods for the HyperDepth implementation


# loads images and extracts their raw feature data
def load_images(file_list, img_path, gt_path, line_idx, displacements, max_radius):
    print("TODO")




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
    df_features = pd.DataFrame(np.reshape(feature_vec[-1, feature_vec.shape[2]]))

    df_features['displc'] = np.reshape(displacement_vec, [-1])
    df_features['xcoord'] = np.reshape(xcoord_vec, [-1])
    df_features['intens'] = np.reshape(intensity_vec, [-1])

    df_features = df_features(df_features['displc'] > 0) # filter out undetermined displacements

    # might need more filtering?

    return df_features

# calculates total signal along a line within a given imgslice
def check_signal(imgslice, radius):
    imgslice = np.float32(imgslice)

    print("TODO")



# extracts sets of feature vectors from an image at predefined coords
def extract_fvecs_image(image, line_idx, displacements, max_radius, n_x):
    print("TODO")

# extracts sets of feature vectors from an NP array of images at predefined coords
def extract_fvecs_array(images, displacements, max_radius, n_x):
    print("TODO")

# generates pre-defined displacement vectors (? do I need this?)
def generate_displacements(num_features, max_radius):
    # might need to handle zero displacement as well
    line_displacements = []

    for disp in range(num_features):
        line_displacements.append(np.random.randint(low = -max_radius, high = max_radius, size = 4)) # size might change
    return line_displacements
    

# calculates pixel difference between displacement vectors along a given line
def extract_line(imgslice, radius, uu, vv):
    print("TODO")

