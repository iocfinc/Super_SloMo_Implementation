import os
from shutil import rmtree, move
import os.path
from pathlib import Path

def main(video_folder):
    test_train_split = (10, 90)
    # Extract video names
    videos = os.listdir(video_folder)
    # Create random train-test split.
    testIndices  = random.sample(range(len(videos)), int((test_train_split[0] * len(videos)) / 100))
    trainIndices = [x for x in range((len(videos))) if x not in testIndices]
    # Create list of video names
    testVideoNames  = [videos[index] for index in testIndices]
    trainVideoNames = [videos[index] for index in trainIndices]
    print("Completed Reading and assigning videos to test and train splits",videos,testIndices, trainIndices, testVideoNames, trainVideoNames, sep='\n')

def create_subfolders(dataset_dest):
    root = Path(dataset_dest)
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

    return extractPath, trainPath, testPath, validationPath 

def extract_frames(testVideoNames,trainVideoNames,video_folder,extractPath, trainPath, testPath, validationPath):
    failed_list = []
    for sublist in [testVideoNames, trainVideoNames]:
        for items in sublist:
            base_dest = Path(extractPath)
            leaf = base_dest/items
            os.makedirs(leaf, exist_ok=True)
            retn = subprocess.call("ffmpeg -i '{}' -vf scale=640:360 -qscale:v 2 '{}}/%04d.jpg'".format(os.path.join(video_folder,items),leaf))
            if retn:
                failed_list.append(items)
    print(failed_list)
                

create_subfolders("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/data")
split("/content/gdrive/My Drive/Colab Notebooks/Super-SloMo/unzipfile/ira")

