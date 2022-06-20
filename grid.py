import cv2
import sys
import pandas as pd
import numpy as np
import imagesize
from tqdm import tqdm
def print_usage():
    print(f"Usage:")
    print(f"python3 {sys.argv[0]} dataframe.csv")



if __name__ == "__main__":
    
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception:
        print_usage()

    # Images per row
    SAMPLES = 5

    labels   = np.unique(df['label'])
    airports = np.unique(df['airport'])
    
    # Loop over each cluster. Each is an image
    for ii, lab in enumerate(tqdm(labels, total=len(labels))):
    
        # Picture the image as a building of many storeys
        image_building = []

        # Loop over each possible airport source - each is a storey
        for jj, air in enumerate(airports):
            try:
                # Sample a few images matching cluster & airport source
                sample_df = df[(df.label==lab) & (df.airport==air)]
                n_matches = len(sample_df)
                images = sample_df.sample(SAMPLES, replace=True)['image']
            except ValueError:
                # Handle no images in that cluster
                image_row = np.hstack([np.zeros((300,300,3))] * SAMPLES)
                image_building.append(image_row)
                continue

            # Build a row of three images
            image_row = []
            for kk, path in enumerate(images):
                img = cv2.imread(path) if isinstance(path,str) else path
                img = cv2.resize(img, (300, 300))
                img = cv2.putText(img, path, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                image_row.append(img)
            image_row = np.hstack(image_row)  
            image_row = cv2.putText(image_row, f'{air}: {n_matches} unique', (20,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) 
            image_row = cv2.putText(image_row, f'{air}: {n_matches} unique', (20,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)  
            image_building.append(image_row)

        image_building = np.vstack(image_building)

        cv2.imwrite(f'out/cluster-{ii}.png', image_building)
