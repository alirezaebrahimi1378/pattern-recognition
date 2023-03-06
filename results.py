import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc , confusion_matrix
import rasterio


# define function to calculate ROC curve
def calculate_roc(mask, ground_truth):
    # calculate false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(ground_truth.ravel(), mask.ravel())
    tn, fp, fn, tp = confusion_matrix(ground_truth.ravel(), mask.ravel()).ravel()
    # calculate area under curve
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc , tp , fp , fn , tn


# load mask and ground truth images
with rasterio.open('/home/alireza/Desktop/final_PR/data/cd_results/CM_0.tif') as mask_src:
    mask = mask_src.read().squeeze()
with rasterio.open('/home/alireza/Desktop/final_PR/data/GT/groundtruth_0.tiff') as gt_src:
    ground_truth = gt_src.read().squeeze()

mask[mask==255] = 1

# calculate ROC curve
fpr, tpr, roc_auc , tp , fp , fn , tn = calculate_roc(mask, ground_truth)
sens = tp/(tp + fn)
TA = (tp + tn)/(tp + tn + fn + fp)
f1 = (2*tp)/(2*tp + fp + fn)
print(f'TA = {TA} / AUC = {roc_auc} / f1 = {f1} / sens = {sens}')
# plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
