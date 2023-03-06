from libsvm.svmutil import  *
import scipy.io
import numpy as np
import random
DATA = scipy.io.loadmat('/home/alireza/Desktop/seg/Indian_pines_corrected.mat')
label = scipy.io.loadmat('/home/alireza/Desktop/seg/Indian_pines_gt.mat')
DATA = DATA['indian_pines_corrected']
lbl = label['indian_pines_gt']
lbl = lbl.flatten()

def flattening(data):
    data = data
    image = np.zeros((data.shape[0]*data.shape[1] , 1))
    for i in range(data.shape[2]):
        band = data[: , : , i]
        band = band.flatten()
        band = band.reshape((-1 , 1))
        image = np.append(image , band , axis=1)
    image = image[: , 1:]
    return image

ind = []
for i in range(20):
    rand = random.randint(0 , 199)
    ind.append(rand)

data = DATA[: , : , ind]
data = flattening(data)
rate = 0.1
k = round(rate * data.shape[0])
test_ind = random.sample(range(0 , data.shape[0]) , k)


test = data[test_ind]
test_gt = lbl[test_ind]
train = np.delete(data , test_ind , axis = 0)
train_gt = np.delete(lbl , test_ind , axis = 0)
print(f'data:"total data : {data.shape}/ test : {test.shape} / train : {train.shape}')

model = svm_train(train_gt , train)
p_labels, p_acc, p_vals = svm_predict(test_gt, test, model)
print(sorted(ind))



