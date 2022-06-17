import cv2
import sys
import pandas as pd
import numpy as np
import imagesize

def print_usage():
    print(f"Usage:")
    print(f"python3 {sys.argv[0]} dataframe.csv")



if __name__ == "__main__":
    
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception:
        print_usage()


    labels   = np.unique(df['label'])
    airports = np.unique(df['airport'])
    
    for ii, lab in enumerate(labels):
        
        # Picture the image as a building of many storeys
        image_building = []
        
        for jj, air in enumerate(airports):
            try:
                sample_df = df[(df.label==lab) & (df.airport==air)]
                sample_df = sample_df.sample(3, replace=True)
                images = sample_df['image']
            except ValueError:
                continue

            # Build a row of three images
            image_row = []
            for kk, path in enumerate(images.values):
                img = cv2.imread(path)
                img = cv2.resize(img, (300, 300))
                image_row.append(img)
            image_row = np.hstack(image_row)    
            image_building.append(image_row)

        image_building = np.vstack(image_building)

        cv2.imshow(f'Cluster {ii}', image_building)
        cv2.waitKey()