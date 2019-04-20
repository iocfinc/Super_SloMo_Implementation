# Creating the Custom Dataset Creator.
# Some changes from the original file is the removal of the `-vsync 0` argument when extracting
# For some reason this causes a failure to fully extract the entire video lengths of the dataset.

import argparse
import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import pickle
from pathlib import Path
from shutil import copyfile, move

# Creating the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--pickle_file", type=str, required=True, default='./Adobe240fps.pkl',help='The path to the pickle file containing the dictionary of Train and Test inputs. Default = Loads the original Adobe240fps dataset.')
parser.add_argument("--base_loc", type=str, required=True, default='./', help='The location of the base directory, where the extracted, train, test and validation folders will be added. Default = Current directory')
parser.add_argument("--input_directory", type=str, required=True, help='The location of the file input videos to be extracted or the unzipped files from the downloaded dataset.')
opts = parser.parse_args()

# Load Pickle file

filename = opts.pickle_file
infile = open(filename, 'rb')
read_dict = pickle.load(infile)
infile.close()

testVideoNames = read_dict['testVideoNames']
trainVideoNames = read_dict['trainVideoNames']

root = Path(opts.base_loc)
if not root.exists():
  os.makedirs(opts.base_loc)

extractPath      = root/ "extracted"
trainPath        = root/ "train"
testPath         = root/ "test"
validationPath   = root/ "validation"

os.makedirs(extractPath, exist_ok = True)
os.makedirs(trainPath, exist_ok = True)
os.makedirs(testPath, exist_ok = True)
os.makedirs(validationPath, exist_ok = True)

failed_list = []
success_list = []
for sublist in [testVideoNames, trainVideoNames]:
#     print("extracting {}".format(item))
    for items in tqdm(sublist, leave = False, desc = 'Extraction progress bar: '):
        base_dest = Path(extractPath)
        leaf = base_dest/items
        os.makedirs(leaf, exist_ok=True)
        leaf.exists()

        retn = os.system("! ffmpeg -i {} -vsync 0 -vf scale=640:360 -qscale:v 2 '{}/%04d.jpg'".format(os.path.join(opts.input_directory,items),leaf))
        if retn:
            failed_list.append(items)
            print("Failed to extract {}".format(items))
        else:
            success_list.append(items)
            print("Extracted {}".format(items))


print('\n*****************************\nFailed to extract: {} \nCompleted extraction: {}'.format(failed_list, success_list))

base_pth = root
extract_pth = base_pth / "extracted"


for folder in [testVideoNames, trainVideoNames]:
    foldercnt = 0
    for sub_folder in folder:
        images = sorted(os.listdir(extract_pth / sub_folder))
        batch_list = [images[x:x+12] for x in range(0, len(images),12)]      
        batch_list = batch_list[:-1] # Do not include the last folder that would be incomplete
        for item in batch_list:
            if (len(item) % 12 ==0):
                if (folder == trainVideoNames):
                    dummy = base_pth / "train" / str(foldercnt)
                else:
                    dummy = base_pth/ "test" / str(foldercnt)
                os.makedirs(dummy, exist_ok= True)
                print('created {}'.format(dummy)) # I was trying to check how many folders are created vs. how many folders reflect
                for filename in item:
                    src = extract_pth / sub_folder/ filename
                    dst = dummy / filename
                    move(src,dst)
                    # print("moving: src {} to  {}".format(src, dst))
                foldercnt +=1
            else:
                break

# images = os.listdir("./data/test")
k = int(0.1*len(images))
to_move = max(1,k)
# k = 10
indices = random.sample(range(len(images)), to_move)

for items in indices:
    src = base_pth / 'test' / str(items)
    dst = base_pth / 'validation' / str(items)
    print('move {} to {}'.format(src, dst))
    copy(src,dst)
    
print('Completed Creating the Test Train Data')
print('Total train: {}\nTotal test: {}\nTotal validation: {}'.format(len(os.listdir(trainPath)),len(os.listdir(testPath)),len(os.listdir(validationPath))))
    