import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rasterio as rio
from libsvm.svmutil import  *
import random

DATA = scipy.io.loadmat('/home/alireza/Desktop/seg/Indian_pines_corrected.mat')
label = scipy.io.loadmat('/home/alireza/Desktop/seg/Indian_pines_gt.mat')
DATA = DATA['indian_pines_corrected']
Lbl = label['indian_pines_gt']
lbl = Lbl.flatten()


def UINT8(Data) :
    shape = Data.shape
    for i in range(shape[2]):
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

def pca_process(data , n_comp=20):
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
data = pca_process(DATA)
plt.subplot(121)
plt.imshow(DATA[: , : , 100] , cmap = "gray")
plt.subplot(122)
plt.imshow(Lbl)
plt.show()
rate = 0.1
k = round(rate * data.shape[0])
test_ind = random.sample(range(0 , data.shape[0]) , k)
test = data[test_ind]
test_gt = lbl[test_ind]
train = np.delete(data , test_ind , axis = 0)
train_gt = np.delete(lbl , test_ind , axis = 0)
print(f'data:"total data : {data.shape[0]}/ test : {test.shape[0]} / train : {train.shape[0]}')

T = [1 , 2 , 3]
kernel = ['polynomial' , 'radial' , 'sigmoid']
for t in T :
    if t == 1 :
        D = np.arange(1 , 3 , 1)
        for d in D:
            test_ind = random.sample(range(0, data.shape[0]), k)
            test = data[test_ind]
            test_gt = lbl[test_ind]
            train = np.delete(data, test_ind, axis=0)
            train_gt = np.delete(lbl, test_ind, axis=0)
            param = f'-h 0 -t {t} -d {d} -q'
            print('*' * 100)
            print(param)
            print(f'using {kernel[t-1]} kernel with degree of {d}')
            model = svm_train(train_gt, train, param)
            p_labels, p_acc, p_vals = svm_predict(test_gt, test, model)
    if t == 2 :
        test_ind = random.sample(range(0, data.shape[0]), k)
        test = data[test_ind]
        test_gt = lbl[test_ind]
        train = np.delete(data, test_ind, axis=0)
        train_gt = np.delete(lbl, test_ind, axis=0)
        print('*' * 100)
        param = f'-h 0 -t {t} -q'
        print(param)
        print(f'using {kernel[t - 1]} with default params')
        model = svm_train(train_gt, train, param)
        p_labels, p_acc, p_vals = svm_predict(test_gt, test, model)
    if t == 3 :
        Coef = np.arange(0 , 100 , 10)
        for coef in Coef :
            test_ind = random.sample(range(0, data.shape[0]), k)
            test = data[test_ind]
            test_gt = lbl[test_ind]
            train = np.delete(data, test_ind, axis=0)
            train_gt = np.delete(lbl, test_ind, axis=0)
            print('*' * 100)
            param = f'-h 0 -r {coef} -q'
            print(param)
            print(f'using {kernel[t - 1]} kernel with coef of {coef}')
            model = svm_train(train_gt, train, param)
            p_labels, p_acc, p_vals = svm_predict(test_gt, test, model)
