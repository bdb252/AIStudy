import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

directory_path = './resMnist'

file_pattern = os.path.join(directory_path, 't10k-*-*.pgm')
pgm_files = glob.glob(file_pattern)
print(pgm_files)

for pgm_files in pgm_files:
    img = Image.open(pgm_files)

    plt.imshow(img, cmap='gray')
    plt.title(f"Image: {pgm_files}")
    plt.axis('off')
    plt.show()