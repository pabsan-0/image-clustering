# Image clustering

Simple K-means clustering of VGG-16 featurized images.


## Usage

- Install requirements and run:
```
$ pip3 install requirements.txt
$ python3 main.py --n-clusters 5 --no-use-cache samples/*.jpg
```

- Find the output in `cache/groups/*.txt`. The you can do some stuff like:
```
# Verify listed items exist
cat cache/groups/0.txt | xargs ls

# Move/link image to pwd
cat cache/groups/0.txt | xargs cp -t .
cat cache/groups/0.txt | xargs realpath | xargs ln -st .
```

- An image-cluster preview is stored in `cache/clusters.png`.

- See available args:

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