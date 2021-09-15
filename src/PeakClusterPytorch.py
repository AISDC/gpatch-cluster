#!/usr/bin/env python

import os
import math 
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
#from sklearn.cluster import KMeans
from libKMCUDA import kmeans_cuda
from kmeans_pytorch import kmeans 
from sklearn.metrics import calinski_harabasz_score
from scipy.interpolate import UnivariateSpline
from torchvision import datasets, models, transforms, utils
from sklearn.preprocessing import StandardScaler

data_dir        = 'data'
num_workers     = 28
crop_size       = 11
data_transforms = transforms.Compose([transforms.CenterCrop(crop_size),
                                      transforms.Grayscale(),
                                      transforms.ToTensor()])
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_dataset   = datasets.ImageFolder(os.path.join(data_dir),data_transforms)
dataloader      = torch.utils.data.DataLoader(image_dataset,num_workers=num_workers,batch_size=len(image_dataset))
images,labels   = next(iter(dataloader))
print(len(image_dataset))

images = images.reshape(len(images),-1)
images = np.array(images) 

clusters=range(3,20)

summed_square_distance=[]
calinski_score=[]

for i in clusters:
    cluster_centers,cluster_ids,average_distance=kmeans_cuda(samples=images,clusters=i,init='random',metric='L2',device=0,seed=1,average_distance=True)#,verbosity=1)
    summed_square_distance.append(average_distance)
    calinski_score.append(calinski_harabasz_score(images,cluster_ids))

#2nd derivative of elbow curve to find optimal number of clusters 
spline    = UnivariateSpline(clusters,calinski_score)
spline_d2 = spline.derivative(n=2) 

d2_list = list(spline_d2(clusters))
idx_max = max(range(len(d2_list)),key=d2_list.__getitem__)
n_clusters=idx_max + min(clusters)
print(n_clusters)

centers,labels = kmeans_cuda(samples=images,clusters=n_clusters,init='random',metric='L2',device=0,seed=1)#,verbosity=1)
Y = labels

plt.figure() 
plt.plot(clusters,d2_list)
yint = range(min(clusters), math.ceil(max(clusters)))
plt.xticks(yint)
plt.savefig('d2_ssd.png') 


#PCA to 2D
pca           = PCA(n_components=2)
pca_transform = pca.fit_transform(images)

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111)
i=0
pca_holder = pd.DataFrame(columns=['pca0','pca1','label'])
pca_holder['pca0']   = pca_transform[:,0]
pca_holder['pca1']   = pca_transform[:,1]
pca_holder['labels'] = Y
unique_labels = set(pca_holder['labels'])
plt.scatter(pca_transform[:,0],pca_transform[:,1], c=Y,cmap=plt.cm.jet)#gist_rainbow)
plt.tight_layout()
plt.savefig('pca_plot.png')

def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = {}
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
    
    return inferred_labels

model = KMeans(n_clusters=n_clusters, random_state=0)
model.fit(images)
Y = model.labels_ 

centroids = model.cluster_centers_
images = centroids.reshape(n_clusters, crop_size, crop_size)
images *= 255
images = images.astype(np.uint8)


# determine cluster labels
cluster_labels = infer_cluster_labels(model, Y)

# create figure with subplots using matplotlib.pyplot
fig, axs = plt.subplots(int(n_clusters/int(np.sqrt(n_clusters))), int(np.sqrt(n_clusters)), figsize = (20, 20))
plt.gray()

# loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):
    
    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))
    
    # add image to subplot
    ax.matshow(images[i])
    ax.axis('off')
    
# display the figure
fig.savefig("centroid_img.png")
