# Image clustering

[TOC]

## Usage

### Locate your images

- Place your images in the `samples/` dir, with `.jpg` entension. This is regex-hardcoded in the scripts.
- RECOMMENDED: **symlink** your images in there instead of copying them:
  - Avoids wasting disk space 
  - Allows you to retain your file tree somewhere else
  - You can fake-change the image extension by renaming just the symlink
- The available `Makefile` automates **my** setup, so you have an idea how to do this. 


### On the go featurizer
This is the simplest way of using this software. For lots of images, we recommend following the steps in the following section instead.

- Run `python3 complete_run.py` to have the images featurized and clustered in one go. 
  - It may take a while. 
  - It will NOT save the featurizer results to a file
  - It will save the results dataframet to the hardcoded path `out/kmeans_run.csv` 



### Featurize then cluster
**Understand** and run these two scripts to first featurize and then cluster those images anytime from the extracted features. Highly recommended if featurizing takes long.

- `featurizer.py`:  `python3 featurizer.py out/features.pickle`
  - Uses a pretrained VGG18 network to batch-featurize images
  - By default, takes the images in `samples/*jpg` (hardcode path inside)
  - Results will be serialized to `out/features.pickle` to a python `dict`:
    ```
    {
        images:     [samples/img1.jpg, ... ,samples/img1000.jpg],
        airports:   [atlas, ... , ilipa],
        features:   [torch.Tensor(*), ... , torch.Tensor(*)]
    }
    ```
  - Respect the extension of the output file: `pickle` or `pkl`. Name/path can be custom.


- `kmeans.py`:  `python3 kmeans.py out/features.pickle [path/out.csv]`
  - Loads the features from `featurizer.py`, stored in `out/features.pickle`.
  - Uses these features to **cluster** the images with standard K-means.
  - Optionally provide `path/out.csv` to store a python `pd.DataFrame` as in:
    ```
    images:     pd.Series[samples/img1.jpg, ... ,samples/img1000.jpg],
    airports:   pd.Series[atlas, ... , ilipa],
    label:      pd.Series[0, ... , 1]
    ```
  - Respect the extension of the input and output files. Output must be `csv`.
  - Adjust the clustering properties by hardcoding the script.


