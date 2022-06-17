import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
import random
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



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
        # Fetch image and
        img_path = self.image_paths[idx]
        airport = img_path.split('/')[-1].split('_')[0]

        # Process the image to match network
        img = cv2.imread(img_path)
        img = self.transform(img)
        img = img.to(self.device)
        return img, img_path, airport

    def __len__(self):
        return len(self.image_paths) 




if __name__ == "__main__":

    # Shuffle and slice a subset for faster testing
    input_images = glob.glob('samples/*.jpg')
    random.shuffle(input_images)
    input_images = input_images[:1000]
    CLUSTERS = 20

    
    ## Featurization of images

    # Deep learning prep junk
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True)
    model = FeatureExtractor(model).to(device)

    # Placeholders to keep track of image-feature-airport triplets
    images = []
    features = []
    airports = []

    # These allow flexible batching
    dataset = MyDataset(input_images, device)
    dataloader = DataLoader(dataset , batch_size=6, shuffle=True)

    # Batch inference
    for img_batch, img_path, airport in tqdm(dataloader):
        with torch.no_grad():
            batch_feat = model(img_batch)

        # De-batch the results to their placeholders
        images.extend(img_path)
        features.extend([bf.cpu().detach().numpy().reshape(-1) for bf in batch_feat])
        airports.extend(airport)



    ## Clustering

    model = KMeans(n_clusters=CLUSTERS, random_state=42)
    model.fit(np.array(features))
    labels = model.labels_



    ## Analysis

    # Dataframe and confusion matrix
    df = pd.DataFrame({'label': labels, 'airport': airports, 'image': images})
    cm = pd.crosstab(df.airport, df.label)

    # Plotting the correlation matrix
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cm, cmap=cmap, linewidths=.5, square=True, annot=True, 
        center=True, annot_kws={"size": 7}, fmt="d")
    
    # Export and go
    df.to_csv('out/kmeans_run.csv')
    plt.show()