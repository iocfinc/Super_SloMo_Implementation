import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import pickle

test_train_split = (10, 90)
# Extract video names from the folder with the videos
# This method assumes that you are using a custom dataset.
# At least have 10 videos for extraction otherwise there would be no test dataset.

videos = os.listdir("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/unzipfile/ira")

# Create random train-test split.
testIndices  = random.sample(range(len(videos)), int((test_train_split[0] * len(videos)) / 100))
trainIndices = [x for x in range((len(videos))) if x not in testIndices]
# Create list of video names
testVideoNames  = [videos[index] for index in testIndices]
trainVideoNames = [videos[index] for index in trainIndices]
print(videos,testIndices, trainIndices, testVideoNames, trainVideoNames, sep='\n')
# Creating a pickle file for the list
indices_dict = {'testVideoNames':testVideoNames, 'trainVideoNames':trainVideoNames}
filename = 'indices'

outfile = open(filename, 'wb')
pickle.dump(indices_dict, filename)
outfile.close()

root = Path("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/data")
if not root.exists():
  os.makedirs("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/data")

extractPath      = root/ "extracted"
trainPath        = root/ "train"
testPath         = root/ "test"
validationPath   = root/ "validation"

os.makedirs(extractPath, exist_ok = True)
os.makedirs(trainPath, exist_ok = True)
os.makedirs(testPath, exist_ok = True)
os.makedirs(validationPath, exist_ok = True)

import subprocess
failed_list = []
success_list = []
for sublist in [testVideoNames, trainVideoNames]:
    print("extracting {}".format(sublist))
    for items in sublist:
        base_dest = Path(extractPath)
        branch = base_dest/sublist
        leaf = branch/items
        os.makedirs(leaf, exist_ok=True)
        leaf.exists()
        print("Extracting: {} in {}".format(items, sublist))
        retn = os.system("ffmpeg -i '{}' -vf scale=640:360 -qscale:v 2 '{}/%04d.jpg'".format(os.path.join("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/unzipfile/ira",items),leaf))
        if retn:
            failed_list.append(items)
            print("Failed to extract {}".format(items))
        else:
            success_list.append(items)
            print("Extracted {}".format(items))
    if (sublist == testVideoNames): # Will Split the files and move to the test folder
      folderCounter = -1
      files = os.listdir()
print("Failed extracting the following items: {} \n Sucessfully extracted the following items: {}".format(failed_list, success_list))

