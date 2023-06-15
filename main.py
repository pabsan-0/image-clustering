import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import List
import shutil


CACHE_IMGS = Path("./cache/images.txt")
CACHE_FEAT = Path("./cache/features.txt")
CACHE_CIMG = Path("./cache/clusters.png")
CACHE_CLST = Path("./cache/groups")


IMAGE_EXTENSIONS = [".jpeg", ".png", ".jpg", ".webp"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-save-cache', type=bool, default=False)
    parser.add_argument('--no-use-cache',  type=bool, default=False)
    parser.add_argument('--no-preview',    type=bool, default=False)
    parser.add_argument('--n-clusters',    type=int, default=3)
    parser.add_argument('unnamed_args', nargs=argparse.REMAINDER)

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
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 


class MyDataset(Dataset):
    def __init__(self , image_paths, device='cpu'):
        """ This dataset class helps fetching each image and its airport
        from a directory while allowing easy batching.
        """
        self.image_paths = image_paths
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor()                              
        ])

    def __getitem__(self, idx): 
        # Fetch image and convert to string
        img_path = self.image_paths[idx]

        # Process the image to match network
        img = cv2.imread(img_path)
        img = self.transform(img)
        img = img.to(self.device)
        return img, img_path

    def __len__(self):
        return len(self.image_paths) 




def featurize(images):
    
    # Deep learning prep junk
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True)
    model = FeatureExtractor(model).to(device)

    # Placeholders to keep track of image-feature triplets
    images = []
    features = []

    # These allow flexible batching
    dataset = MyDataset(input_images, device)
    dataloader = DataLoader(dataset , batch_size=6, shuffle=True)

    # Batch inference
    for img_batch, img_path in tqdm(dataloader):
        with torch.no_grad():
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


def np_loadtxt_or_none(path):
    try:
        return np.loadtxt(path, dtype=object)
    except FileNotFoundError:
        return np.array([])



def build_clustered_img(cluster_list: List[List[str]]):
    DISPLAY_COLS = 5
    DISPLAY_IM_H = 300
    DISPLAY_IM_W = 400
    DISPLAY_GAP = 10

    canvas = []
    canvas.append(np.ones([DISPLAY_GAP, DISPLAY_COLS * (DISPLAY_IM_W + DISPLAY_GAP), 3]))
    for ii, clus in enumerate(cluster_list):
        
        clus = random.sample(clus, min(len(clus), DISPLAY_COLS))
        while len(clus) < DISPLAY_COLS:
            clus.append(np.ones([DISPLAY_IM_H, DISPLAY_IM_W, 3]) * 255)
        
        clus_imdata = []
        for path in clus:
            img = cv2.imread(path) if isinstance(path,str) else path
            path = path if isinstance(path, str) else ""
            img = cv2.resize(img, (DISPLAY_IM_W, DISPLAY_IM_H))
            img = cv2.putText(img, str(ii) + path, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            clus_imdata.append(img)
            clus_imdata.append(np.ones([DISPLAY_IM_H, DISPLAY_GAP, 3]) * 255)

        globals().update(locals())
        canvas.append(np.hstack(clus_imdata))
        canvas.append(np.ones([DISPLAY_GAP, DISPLAY_COLS * (DISPLAY_IM_W + DISPLAY_GAP), 3]))

    canvas = np.vstack(canvas)
    return canvas



if __name__ == "__main__":
    args = parse_args()

    input_images = [ii for ii in args.unnamed_args if Path(ii).suffix in IMAGE_EXTENSIONS]
    random.shuffle(input_images)
    
    
    # Compute features or load from cache
    cached_images = np_loadtxt_or_none(CACHE_IMGS)
    cached_features = np_loadtxt_or_none(CACHE_FEAT)
    if set(input_images) == set(cached_images) and not args.no_use_cache:
        images, features = cached_images, cached_features
    else:
        images, features = featurize(input_images)
   

    # Save features if not told otherwise only if they are new
    if not (args.no_save_cache or features is cached_features):
        CACHE_CLST.mkdir(parents=True, exist_ok=True)
        np.savetxt(CACHE_FEAT, features, fmt='%s')
        np.savetxt(CACHE_IMGS, images, fmt='%s')


    
    num_clusters = args.n_clusters
    while True:

        # Create dump dir for clusters
        shutil.rmtree(CACHE_CLST, ignore_errors=True)
        CACHE_CLST.mkdir(parents=True, exist_ok=True)

        group, image, cluster_list = cluster(images, features, n=num_clusters)
        
        # Groups output
        for ii, clus in enumerate(cluster_list):
            np.savetxt(CACHE_CLST / (str(ii) + ".txt"), clus, fmt='%s')
        
        # Preview output
        if not args.no_preview:
            preview = build_clustered_img(cluster_list)
            cv2.imwrite(CACHE_CIMG.as_posix(), preview)
            print("Image saved to", CACHE_CIMG)
        
        # Close loop
        try:
            num_clusters = int(input("Try a different number of clusters? >> "))
        except EOFError as e:
            print()
            exit()
        except Exception as e:
            print(e)
            print("Wrong input!")
                
            


