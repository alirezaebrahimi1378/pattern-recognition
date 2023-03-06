import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
path = '/home/alireza/Desktop/seg/rectangle3.tif'
def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[0]):
        data = Data[i , : , :]
        data = data / data.max()
        data = 255 * data
        Data[i] = data.astype(np.uint8)
    return Data

def Uint8(img):
    img = img/img.max()
    img = img * 255
    img = img.astype(np.uint8)
    return img

def pca_process(data , n_comp=3):
    data = data.transpose(1, 2, 0)
    image = np.zeros((data.shape[0]*data.shape[1] , 1))
    for i in range(data.shape[2]):
        band = data[: , : , i]
        band = StandardScaler().fit_transform(band)
        band = band.flatten()
        band = band.reshape((-1 , 1))
        image = np.append(image , band , axis=1)
    image = image[: , 1:]
    pca = PCA(n_components=n_comp)
    pca_img = pca.fit_transform(image)
    return pca_img

image = rio.open(path)
img = image.read()
img = UINT8(img)
img_pca = pca_process(img)
img_vis1 = img[1:4, : , :]
img_vis1 = np.flip(img_vis1 , 0)
img_vis1 = img_vis1.transpose(1 , 2 , 0)

# plt.subplot(1 , 4 ,1)
# plt.imshow(img_vis1.astype('uint8'))
# plt.title('original image')

plt.subplot(1 , 3 , 1)
img_pca1 = img_pca[: , 0]
img_pca1 = img_pca1.reshape(img_vis1.shape[0] , img_vis1.shape[1])
plt.imshow(img_pca1, cmap = 'gray')
plt.title('pca_1')

plt.subplot(1 , 3 , 2)
img_pca2 = img_pca[: , 1]
img_pca2 = img_pca2.reshape(img_vis1.shape[0] , img_vis1.shape[1])
plt.imshow(img_pca2, cmap = 'gray')
plt.title('pca_2')

plt.subplot(1 , 3 , 3)
img_pca3 = img_pca[: , 2]
img_pca3 = img_pca3.reshape(img_vis1.shape[0] , img_vis1.shape[1])
plt.imshow(img_pca3, cmap = 'gray')
plt.title('pca_3')

plt.show()