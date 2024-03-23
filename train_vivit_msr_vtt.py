import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
#!pip install einops icecream
from tqdm import tqdm
from msr_vtt_frames import MSRDataset
from vivit_model1 import ViViT_2
import torchvision
from einops import rearrange
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
device ='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

frames_per_clip = 32
batch_size = 1
dataset_dir = 'test_1k_msrvtt'

num_classes = 20

train_val_data = MSRDataset( dataset_dir = dataset_dir, subset="train", video_list_file="msrvtt_names_test.csv",frames_per_clip=frames_per_clip)
print("Length of train data", len(train_val_data))
train_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True)


model = ViViT_2(image_size=224, patch_size=16, num_classes=num_classes, frames_per_clip=frames_per_clip, tube= True)
checkpoint = torch.load('pos16.pt',map_location=torch.device(device))
unmatched = model.load_state_dict(checkpoint, strict= False)

for i in unmatched.missing_keys:
    print(i)


model.to(device)
model.eval()
features = []

with open('data/msrvtt/clip/msrvtt_jsfusion_test.pkl', 'rb') as f:
    # Load the contents of the pickle file
    data = pickle.load(f)

selected_vids = []




with torch.no_grad():
    for batch_id,(video,name) in tqdm(enumerate(train_loader)):
        video = video.to(device)
        pred = model(video)
        pred= pred
        pred = pred.cpu().numpy()
        #pred = dim_reduce(pred)
        if batch_id<=6783:
            for i in data:
                i['clip'] = pred
        #features.append(pred)

with open('data/msrvtt/clip/msrvtt_test_vivit.pkl', 'wb') as file:
    pickle.dump(data, file)
print("done")








