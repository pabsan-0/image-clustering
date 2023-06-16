import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import List
import shutil
import cv2
    

CACHE_FEAT_IMAGES = Path("./cache/images.txt")
CACHE_FEAT_FEATURES = Path("./cache/features.txt")
CACHE_CLUS_PREVIEW = Path("./cache/clusters.png")
CACHE_CLUS_GROUPS = Path("./cache/groups")

FEATURIZER_IMSIZE = 448
FEATURIZER_NUM_WORKERS = torch.multiprocessing.cpu_count()

PREVIEW_COLS = 5
PREVIEW_IM_H = 300
PREVIEW_IM_W = 400
PREVIEW_GAP = 10

ALLOWED_IMAGE_EXTENSIONS = [".jpeg", ".png", ".jpg", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n-clusters', type=int, default=3,  help="K means number of clusters")
    parser.add_argument('-b', '--batch-size', type=int, default=6,  help="Batch size for VGG16 inference")
    parser.add_argument('--no-save-cache',    action='store_true',  help="Do not cache features")
    parser.add_argument('--no-use-cache',     action='store_true',  help="Do not load features from cache")
    parser.add_argument('--no-preview',       action='store_true',  help="Do not build a preview image")
    parser.add_argument('images', nargs=argparse.ONE_OR_MORE, help="Images to cluster, accepts a glob")

    return parser.parse_args()



class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()

        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)

        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool

        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()

        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
    
    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 


class MyDataset(Dataset):
    def __init__(self , image_paths):
        """ This dataset class helps fetching each image from a directory while
        allowing easy transforms and batching.
        """
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(FEATURIZER_IMSIZE),
            transforms.CenterCrop(FEATURIZER_IMSIZE),
            transforms.ToTensor()                              
        ])

    def __getitem__(self, idx): 
        # Fetch image and convert to string
        img_path = self.image_paths[idx]

        # Process the image to match network
        img = cv2.imread(img_path)
        img = self.transform(img)
        # Only CPU operations here so I can use multiprocessing dataloaders
        return img, img_path

    def __len__(self):
        return len(self.image_paths) 


class MyUtils:

    @staticmethod
    def np_loadtxt_or_none(path):
        try:
            return np.loadtxt(path, dtype=object)
        except FileNotFoundError:
            return np.array([])


    @staticmethod
    def build_clustered_img(cluster_list: List[List[str]]):
        canvas = []
        canvas.append(np.ones([PREVIEW_GAP, PREVIEW_COLS * (PREVIEW_IM_W + PREVIEW_GAP), 3]))

        # Each iteration processes a whole cluster, displayed horizontally
        for ii, clus in enumerate(cluster_list):
            
            # Draw samples for the preview and fill with blanks if not enough
            clus_sampled = random.sample(clus, min(len(clus), PREVIEW_COLS))
            while len(clus_sampled) < PREVIEW_COLS:
                clus_sampled.append(np.ones([PREVIEW_IM_H, PREVIEW_IM_W, 3]) * 255)
            
            # Each iteration adds a sample from this cluster
            clus_imdata = []
            for path in clus_sampled:
                img = cv2.imread(path) if isinstance(path,str) else path
                path = path if isinstance(path, str) else ""
                img = cv2.resize(img, (PREVIEW_IM_W, PREVIEW_IM_H))
                img = cv2.putText(img, str(ii) + path, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                clus_imdata.append(img)
                clus_imdata.append(np.ones([PREVIEW_IM_H, PREVIEW_GAP, 3]) * 255)

            # Join this cluster horizontally w/ a gap, write its number of instances
            clus_imdata = np.hstack(clus_imdata)
            clus_imdata = cv2.putText(clus_imdata, str(len(clus)), (20, PREVIEW_IM_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (1,1,1),       20)
            clus_imdata = cv2.putText(clus_imdata, str(len(clus)), (20, PREVIEW_IM_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 10)
            canvas.append(clus_imdata)
            canvas.append(np.ones([PREVIEW_GAP, PREVIEW_COLS * (PREVIEW_IM_W + PREVIEW_GAP), 3]))

        # Join all cluster rows vertically    
        canvas = np.vstack(canvas)
        return canvas




def featurize(images, batch_size=6):
    
    # Deep learning prep junk
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True)
    model = FeatureExtractor(model).to(device)

    # Placeholders to keep track of image-feature triplets
    images = []
    features = []

    # These allow flexible batching
    dataset = MyDataset(input_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
        num_workers=FEATURIZER_NUM_WORKERS
    )

    # Batch inference
    for img_batch, img_path in tqdm(dataloader):
        with torch.no_grad():
            # Send to GPU here so I can use multiprocessing dataloaders
            img_batch = img_batch.to(device, non_blocking=True)
            batch_feat = model(img_batch)

        # De-batch the results to their placeholders
        images.extend(img_path)
        features.extend([bf.cpu().detach().numpy().reshape(-1) for bf in batch_feat])


    return np.array(images), np.array(features)



def cluster(images: np.array, features: np.array, n=10):
    model = KMeans(n_clusters=n, random_state=42)
    model.fit(features)
    groups = np.array(model.labels_)
    
    cluster_list = []
    df_groups = pd.DataFrame({'groups': groups, 'images': images})
    for ii in range(num_clusters):
        clus = df_groups[df_groups['groups']==ii]["images"].tolist()
        cluster_list.append(clus)

    return groups, images, cluster_list



if __name__ == "__main__":
    args = parse_args()

    assert args.images, "Positional argument `images` was not provided."
    input_images = [ii for ii in args.images if Path(ii).suffix in ALLOWED_IMAGE_EXTENSIONS]
    random.shuffle(input_images)

    # Compute features or load from cache
    cached_images = MyUtils.np_loadtxt_or_none(CACHE_FEAT_IMAGES)
    cached_features = MyUtils.np_loadtxt_or_none(CACHE_FEAT_FEATURES)
    if set(input_images) == set(cached_images) and not args.no_use_cache:
        images, features = cached_images, cached_features
    else:
        images, features = featurize(input_images, args.batch_size)


    # Save features if not told otherwise only if they are new
    if not (args.no_save_cache or features is cached_features):
        CACHE_CLUS_GROUPS.mkdir(parents=True, exist_ok=True)
        np.savetxt(CACHE_FEAT_FEATURES, features, fmt='%s')
        np.savetxt(CACHE_FEAT_IMAGES, images, fmt='%s')


    
    num_clusters = args.n_clusters
    while True:

        # Create dump dir for clusters
        shutil.rmtree(CACHE_CLUS_GROUPS, ignore_errors=True)
        CACHE_CLUS_GROUPS.mkdir(parents=True, exist_ok=True)

        group, image, cluster_list = cluster(images, features, n=num_clusters)
        
        # Groups output
        for ii, clus in enumerate(cluster_list):
            np.savetxt(CACHE_CLUS_GROUPS / (str(ii) + ".txt"), clus, fmt='%s')
        
        # Preview output
        if not args.no_preview:
            preview = MyUtils.build_clustered_img(cluster_list)
            cv2.imwrite(CACHE_CLUS_PREVIEW.as_posix(), preview)
            print("Image saved to", CACHE_CLUS_PREVIEW)
        
        # Close loop
        try:
            num_clusters = int(input("Try a different number of clusters? >> "))
        except EOFError as e:
            print()
            exit()
        except Exception as e:
            print(e)
            print("Wrong input!")
                
            


