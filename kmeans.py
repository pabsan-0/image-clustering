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
from torch.utils.data import Dataset

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



# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()                              
])

CLUSTERS = 20
air2int = {
    'aerohispalis': 0,
    'beas': 1,
    'oran': 2,
    'atlas': 3,
    'ilipa': 4,
}


if __name__ == "__main__":

    # Sort gpu/cpu and boot the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True)
    model = FeatureExtractor(model).to(device)


    # Placeholders, notice their sorting matches
    images = glob.glob('samples/*.jpg')
    features = []
    airports = []

    random.shuffle(images)
    images = images[:20]

    # Featurization of images
    for img_path in tqdm(images):
        airport = img_path.split('/')[-1].split('_')[0]
        airports.append(airport)

        # Read and apply transformations
        img = cv2.imread(img_path)
        img = transform(img)

        # [batch_size, channels, width, height]
        img = img.reshape(1, 3, 448, 448)
        img = img.to(device)

        # We only extract features, so we don't need gradient
        with torch.no_grad():
            features.append(model(img).cpu().detach().numpy().reshape(-1))

    # Convert to NumPy Array
    features = np.array(features)

    # Initialize the model
    model = KMeans(n_clusters=CLUSTERS, random_state=42)
    model.fit(features)
    labels = model.labels_ # [4 3 3 ... 0 0 0]

    # Dataframe and confusion matrix
    df = pd.DataFrame({'label': labels, 'airport': airports, 'image': images})
    cm = pd.crosstab(df.airport, df.label)

    # Plotting the correlation matrix
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cm, cmap=cmap, linewidths=.5, square=True, annot=True, 
        center=True, annot_kws={"size": 7}, fmt="d")
    
    # Export and go
    df.to_csv('filename.csv')
    plt.show()