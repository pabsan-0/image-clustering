# Image clustering

Simple K-means clustering of VGG-16 featurized images.


## Requirements

- Install through pip: `$ pip3 install requirements.txt`
- The following are also needed, yet commented out in the requirements file not to mess with your versions:
    - Pytorch
    - Opencv


## Usage

- Run featurization & clustering with: 
```
$ python3 main.py --n-clusters 5 --no-use-cache samples/*.jpg
```
- An image-cluster preview is stored in `cache/clusters.png`.
- Find the output in `cache/groups/*.txt` as path lists of your files by cluster. The you can do some stuff like:
```
# Verify listed items exist
cat cache/groups/0.txt | xargs ls

# Move/link image to pwd
cat cache/groups/0.txt | xargs cp -t .
cat cache/groups/0.txt | xargs realpath | xargs ln -st .
```

By default, intermediate files containing the list of input images and their features will be stored on disk. If trying to re-featurize the very same images, cached features will be loaded to save time.

## Scripts

```
$ python3 main.py --help 
usage: main.py [-h] [-n N_CLUSTERS] [--no-save-cache] [--no-use-cache] [--no-preview] ...

positional arguments:
  images                Images to cluster, accepts a glob

optional arguments:
  -h, --help            show this help message and exit
  -n N_CLUSTERS, --n-clusters N_CLUSTERS
                        K means number of clusters
  --no-save-cache       Do not cache features
  --no-use-cache        Do not load features from cache
  --no-preview          Do not build a preview image
```


## Known issues

- `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling 'cublasCreate(handle)'`
    - Random error, just rerun
    - If persisting, try decreasing batch size in the dataloader func


## Diving deeper

- https://datagen.tech/guides/computer-vision/vgg16/
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html