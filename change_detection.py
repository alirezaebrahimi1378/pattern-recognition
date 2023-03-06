import cv2
import numpy as np
import rasterio as rio
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
from utils_KMEANS import KMEANS
from skimage.exposure import match_histograms
import warnings

################################# functions #################################

def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[2]):
        data = Data[: , : , i]
        data = data / data.max()
        data = 255 * data
        Data[: , : ,i] = data.astype(np.uint8)
    return Data

def get_index(path):
    path = path.replace("/" , " ")
    path = path.replace("_" , " ")
    path = path.replace("." , " ")
    return path.split()[8]

################################# main code #################################
warnings.filterwarnings("ignore")
image1_paths = []
image2_paths = []
# finding all images in directory
def change_detection(root_dir) :
    input_dir = os.path.join(root_dir, 'data/date1')
    output_dir = os.path.join(root_dir, 'data/cd_results')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for root, dirs, files in os.walk(input_dir , topdown = True):
        for i in range(len(files)):
            address = root + '/' + files[i]
            image1_paths.append(address)
            ind = get_index(address)
            address = f'{root_dir}/data/date2/sentinel2_{ind}.tiff'
            image2_paths.append(address)

    for i in range(len(image1_paths)):
        image1_path = image1_paths[i]
        ind = get_index(image1_path)
        image2_path = image2_paths[i]
        img1 = rio.open(image1_path)
        transform = img1.transform
        crs = img1.crs
        shape = img1.shape
        image1 = img1.read()
        img2 = rio.open(image2_path)
        image2 = img2.read()
        image1 = image1.transpose(1, 2, 0)
        image2 = image2.transpose(1, 2, 0)
        #########
        image1 = match_histograms(image1, image2, channel_axis=-1)
        img1_vis = image1[:, :, 10:13].astype(np.uint8)
        img2_vis = image2[:, :, 10:13].astype(np.uint8)

        image1 = UINT8(image1).astype(float)
        image2 = UINT8(image2).astype(float)
        # resizing image for k-means algorithm
        x , y = image1.shape[0], image1.shape[1]
        new_sizex = np.asarray(image1.shape[0]) / 5
        new_sizex = new_sizex.astype(int) * 5
        new_sizey = np.asarray(image1.shape[1]) / 5
        new_sizey = new_sizey.astype(int) * 5
        new_size = [new_sizey, new_sizex]
        image1_new = resize(image1, (new_sizex, new_sizey, image1.shape[2])).astype(int)
        image2_new = resize(image2, (new_sizex, new_sizey, image1.shape[2])).astype(int)

        ########### k-means_PCA CD ###########
        comp = 3                             # number of clusters
        bands = 3                            # number of bands
        change_mask = KMEANS(image1_new , image2_new , new_size , comp , bands , use_all_bands=True)
        change_mask_kmeans = cv2.resize(change_mask, (y, x) ,interpolation= cv2.INTER_NEAREST ).astype(int)
        os.chdir(output_dir)
        new_dataset = rio.open(f"CM_{ind}.tif", 'w', driver='GTiff',
                               height=shape[0], width=shape[1],
                               count=1, dtype=str('uint8'),
                               crs=crs,
                               transform=transform)
        new_dataset.write(np.array([change_mask_kmeans]))
        new_dataset.close()
        plt.figure(figsize=(20 , 20))
        ax = plt.subplot(131)
        plt.imshow(img1_vis)
        plt.title('image1')
        plt.subplot(132, sharex = ax , sharey = ax)
        plt.imshow(img2_vis)
        plt.title('image2')
        plt.subplot(133, sharex = ax , sharey = ax)
        plt.imshow(change_mask_kmeans , cmap = 'gray')
        plt.title('kmens')
        plt.show()
root = os.getcwd()
change_detection(root)