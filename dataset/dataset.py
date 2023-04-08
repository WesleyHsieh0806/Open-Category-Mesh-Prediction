import torch
from torch.utils.data import Dataset, DataLoader


import pytorch3d
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat


# Python 3.9
# conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install pytorch3d -c pytorch3d



import glob
import json
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np


import cv2


IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])


ORIGIN_IMAGE_SIZE  = (1080, 1920, 3)
OUTPUT_IMAGE_SIZE = (144, 256, 3)


io = IO()
io.register_meshes_format(MeshGlbFormat())



def readImage(fn):
    img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img, OUTPUT_IMAGE_SIZE[:2][::-1]).astype(float) / 255.0
    norm_img = (resized_img - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(norm_img).float()




def readGLBToMesh(fn):
    mesh = io.load_mesh(fn, include_textures=False, device="cpu")
    return mesh



class ObjaverseDataset(Dataset):

    def __init__(self, args):

        self.args = args
        # Path to Objaverse        
        self.data_root = args["data_root"] #args.data_root

        # Path to the json files for dataset splitting
        self.split_data_path = args["split_data_path"] #args.split_data_path
        
        # Path of all uid folder
        self.all_path_prefix = []
        
        # All image (Normalized following the ImageNet)
        self.all_img = None
        
        # All target mesh
        self.all_mesh = None

        # All corresponding categoty label
        self.all_label = []
        self.label2id = {}

        # Load all the data and mesh into RAM
        self.loadData()

    def loadData(self):
        
        print("Building Dataset from {}".format(self.split_data_path.split(os.sep)[-1]))
        with open(self.split_data_path, "r") as fd:
            cate2uid = json.load(fd)
        
        # Formulate the path prefix for each data
        for c in cate2uid:
            for uid in cate2uid[c]:
                path_prefix = os.path.join(self.data_root, "data", c, uid)
                self.all_path_prefix.append(path_prefix)
                self.all_label.append(c)

            self.label2id[c] = len(self.label2id)
        

        # Read All Image
        print("Reading All Image ...")
        self.all_img = Parallel(n_jobs=self.args["num_worker"])(
            delayed(readImage)(os.path.join(p, "image.jpeg")) for p in tqdm(self.all_path_prefix)
        )

        # Read All 3d Object
        print("Reading All Mesh ...")
        self.all_mesh = Parallel(n_jobs=self.args["num_worker"])(
                delayed(readGLBToMesh)(os.path.join(p, "3Dobject.glb")) for p in tqdm(self.all_path_prefix)
        )

        # Testing
        # no_mesh_data = []
        # for p in tqdm(self.all_path_prefix):
        #     try:
        #         mesh = readGLBToMesh(os.path.join(p, "3Dobject.glb"))
        #     except:
        #         no_mesh_data.append(p)
        # print(no_mesh_data)

    def __getitem__(self, idx):
        return self.all_img[idx], self.all_mesh[idx]

    def __len__(self):
        return len(self.all_path_prefix)


# Input:
# Output: Dict
def collate_batched(data):
    img, mesh = zip(*data)

    batched_img_ten = torch.stack(img).permute(0,3,1,2)
    batched_mesh_ten = pytorch3d.structures.join_meshes_as_batch(mesh)

    out_dict = {
        "img": batched_img_ten, 
        "mesh": batched_mesh_ten
    }

    return out_dict




def get_dataloader(args):
    
    dataset = ObjaverseDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=args["train"],
        num_workers=args["num_worker"],
        collate_fn=collate_batched
    )

    return dataset, dataloader



if __name__ == "__main__":

    args = {
        "data_root": "/media/godel/HDD/jimmy/Objaverse/",
        "num_worker": 8,
        "split_data_path": "/home/jimmy/HumanPose/NovelObject/Open-Category-Mesh-Prediction/dataset/train_seen_uid.json",
        "train": False,
        "batch_size": 16
    }

    train_dataset, train_dataloader = get_dataloader(args)




# Non Triangular Mesh
# "wolf": "eab21bf368fa4abcaf1b01c5abc4d8ea",
# /media/godel/HDD/jimmy/Objaverse/data/gargoyle/769829932659490c936ab574af87c523
# /media/godel/HDD/jimmy/Objaverse/data/bookcase/bc00a1bb133a43d1a2bfa2b2cb4c94f9
# /media/godel/HDD/jimmy/Objaverse/data/bus_vehicle/ca65afbb228f4af09b98bb19a15de304
# /media/godel/HDD/jimmy/Objaverse/data/bus_vehicle/c42b7bae122e421fb0c080a49178e498
# /media/godel/HDD/jimmy/Objaverse/data/bus_vehicle/c8dc1a6b6f1849228ec603f3b341b2a0
# /media/godel/HDD/jimmy/Objaverse/data/fighter_jet/5669fb9a087d434bafa9bc127aedf006
# /media/godel/HDD/jimmy/Objaverse/data/fighter_jet/b02dcfe9b3ef4cae9251fda9a83651da
# /media/godel/HDD/jimmy/Objaverse/data/desk/b7a9b0e884c849d3a0b93a3a6aa85bf3
# /media/godel/HDD/jimmy/Objaverse/data/penguin/c246565eb927410486c7cf27b138a2e2
# /media/godel/HDD/jimmy/Objaverse/data/lamppost/df09f51886824cd78957d92017fef972
# /media/godel/HDD/jimmy/Objaverse/data/lamppost/e937afc06a1643f3bac57da9aec9e308
# /media/godel/HDD/jimmy/Objaverse/data/lamppost/ad9e0859b1b44adf950695ead8c02a9a
# /media/godel/HDD/jimmy/Objaverse/data/lamppost/9b295aeacd7549d5a09f6e051a4712a1


# /media/godel/HDD/jimmy/Objaverse/data/horse/36ec2a3c75ed42c3aa673597dec72a1a
# /media/godel/HDD/jimmy/Objaverse/data/horse/ae10ee3022bd4212970c723ffcae71fb
# /media/godel/HDD/jimmy/Objaverse/data/award/ff7b5ec543934e32b94d4dcd729aa822
# /media/godel/HDD/jimmy/Objaverse/data/award/7b07838985a947fda70036aafd46b3ef
# /media/godel/HDD/jimmy/Objaverse/data/award/1006883bdd9e4fb5a7321e654d4046dd
# /media/godel/HDD/jimmy/Objaverse/data/award/8e92dd7490484eb3ab615f99a414bf55
# /media/godel/HDD/jimmy/Objaverse/data/lion/92119cd309ac4d629f8f8da5803d98ed
# /media/godel/HDD/jimmy/Objaverse/data/armoire/3597a7a470ac4f81b1b362eea66f4d6f
# /media/godel/HDD/jimmy/Objaverse/data/armoire/757961eb75a64dc689f2047ae8cdbd3b
# /media/godel/HDD/jimmy/Objaverse/data/bullet_train/cfa917f046264e7bab595311730bf097
# /media/godel/HDD/jimmy/Objaverse/data/shark/9fc13419b7df453d8dadd1437381d390
# /media/godel/HDD/jimmy/Objaverse/data/piano/dd034472beb1473ab94544f5acf0fd91
# /media/godel/HDD/jimmy/Objaverse/data/antenna/f5752293425f453d8277657e73f8bbe9
# /media/godel/HDD/jimmy/Objaverse/data/antenna/e1349e9e905a4587b816f327953debc7
# /media/godel/HDD/jimmy/Objaverse/data/antenna/5bffc51e8d084b31b191640f665b8094
# /media/godel/HDD/jimmy/Objaverse/data/antenna/c2cf3c69a2b74f97941a9072334e08c5
# /media/godel/HDD/jimmy/Objaverse/data/antenna/526d8fefa4654b968347d8df044d0c45
# /media/godel/HDD/jimmy/Objaverse/data/antenna/



# ['/media/godel/HDD/jimmy/Objaverse/data/bus_vehicle/5296065cb6d942ad9eb79f4f44cb4526', 
#  '/media/godel/HDD/jimmy/Objaverse/data/lamppost/c8cee098f2cc46b4b30062aa459651a2', 
#  '/media/godel/HDD/jimmy/Objaverse/data/lamppost/9212492648e643cbb34d52b97eeb5587']


# '/media/godel/HDD/jimmy/Objaverse/data/penguin/466390ba903248baa31a3c0977666a95'


# ['/media/godel/HDD/jimmy/Objaverse/data/earphone/538487c7b2f949f0b45ed438923941a9', 
#  '/media/godel/HDD/jimmy/Objaverse/data/armoire/1a41218e823c4cba801bd2b3d71c9dff']