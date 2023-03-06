from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm



def find_vector_set(diff_image, new_size):
    i = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        j = 0
        while j < new_size[1]:
            k = 0
            while k < new_size[0]:
                block = diff_image[j:j + 5, k:k + 5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
                i = i + 1
            j = j + 5


    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):
    i = 2
    feature_vector_set = []
    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = diff_image[i - 2:i + 3, j - 2:j + 3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    return FVS

def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)
    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (new[1] - 4, new[0] - 4))
    return least_index, change_map


def KMEANS(image1 ,image2 , new_size , comp=5 , classes=12 , use_all_bands = True):

    diff_image = abs(image1 - image2)
    change_mask = np.zeros((diff_image.shape[0] - 4, diff_image.shape[1] - 4))

    if use_all_bands == True:
        for m in tqdm(range(diff_image.shape[2])):
            diff = diff_image[:, :, m]

            pca = PCA()
            vector_set, mean_vec = find_vector_set(diff, new_size)
            pca.fit(vector_set)
            EVS = pca.components_

            FVS = find_FVS(EVS, diff, mean_vec, new_size)
            components = comp
            least_index, change_map = clustering(FVS, components, new_size)

            change_map[change_map == least_index] = 255
            change_map[change_map != 255] = 0
            change_map = change_map.astype(np.uint8)
            change_mask = change_mask + change_map

        change_mask = np.where(change_mask > classes * 255, 255, 0)


    if use_all_bands == False :
        diff = diff_image[:, :, 11]
        pca = PCA()
        vector_set, mean_vec = find_vector_set(diff, new_size)
        pca.fit(vector_set)
        EVS = pca.components_

        FVS = find_FVS(EVS, diff, mean_vec, new_size)
        components = comp
        least_index, change_map = clustering(FVS, components, new_size)

        change_map[change_map == least_index] = 255
        change_map[change_map != 255] = 0
        change_mask = change_map.astype(np.uint8)

    return (change_mask)