
import torchvision as tv
import torch
from torch.nn import functional as F

#!pip install einops icecream
import decord
import numpy as np
from PIL import Image
from icecream import ic
import pandas as pd

class MSRDataset(torch.utils.data.Dataset):
    """
    Dataset Class for reading UCF101 dataset  
    
    Args:
        dataset_dir: (str) - root directory of dataset
        subset: (str) - train or test subset
        video_list_file: (str) - file name containing list of video names 
        frames_per_clip: (int) - number of frames to be read in every video clip [default:16]
    """

    #class_names = [x.strip().split()[1] for x in open('UCF101/classInd.txt').readlines()]
    def __init__(self, dataset_dir, subset, video_list_file, frames_per_clip=16):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset=subset
        self.video_list_file = pd.read_csv(dataset_dir+'/'+ video_list_file)

        if self.subset=="train":
            self.video_list = self.video_list_file['video_path'].tolist()
        else:
            #self.video_list = [files[:-1] for files in video_names_file.readlines()]
            with open(f'{dataset_dir}/classInd.txt') as self.classIndices:
                values,keys=zip(*(files[:-1].split() for files in self.classIndices.readlines()))
                self.indices = dict( (k,v) for k,v in zip(keys,values))

        self.frames_per_clip = frames_per_clip

        self.transform = tv.transforms.Compose([
          tv.transforms.Resize(256),
          tv.transforms.CenterCrop(224), # (224x224)
          tv.transforms.ToTensor(),
          tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # (R,G,B) (mean, std)
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        videoname = f'{self.video_list[idx]}.mp4'
        vid = decord.VideoReader(f'{self.dataset_dir}/{videoname}', ctx=decord.cpu(0)) # for reading frames in videos
        nframes = len(vid)

        # if number of frames of video is less than frames_per_clip, repeat the frames
        if nframes <= self.frames_per_clip:
            idxs = np.arange(0, self.frames_per_clip).astype(np.int32)
            idxs[nframes:] %= nframes

        # else if frames_per_clip is greater, sample uniformly seperated frames
        else:
            idxs = np.linspace(0, nframes-1, self.frames_per_clip)
            idxs = np.round(idxs).astype(np.int32)

        imgs = []
        for k in idxs:
            frame = Image.fromarray(vid[k].asnumpy())
            frame = self.transform(frame)
            imgs.append(frame)
        imgs = torch.stack(imgs)

        # if its train subset, return both the frames and the label 
        #if self.subset=="train":
            #label = int(self.labels[idx]) - 1    
        # else, for test subset, read the label index
        #else:
            #label=int(self.indices[videoname.split('/')[2]])-1
        return imgs,videoname  #,label

#Dataset = MSRDataset('TrainValVideo', 'train', 'msrvtt_names.csv') 
#print(Dataset[0].shape)



