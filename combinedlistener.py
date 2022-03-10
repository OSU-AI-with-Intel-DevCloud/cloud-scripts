import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

frame_h = 5
frame_l = 5

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU:", gpu)

import sys
sys.path.insert(0, "input/blazeface-pytorch")
sys.path.insert(0, "input/deepfakes-inference-demo")

from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)

from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = frame_h * frame_l
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

input_size = 224

from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

import torch.nn as nn
import torchvision.models as models

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)
        
checkpoint = torch.load("input/deepfakes-inference-demo/resnext.pth", map_location=gpu)

model = MyResNeXt().to(gpu)
model.load_state_dict(checkpoint)
_ = model.eval()

del checkpoint

def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                    
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            if n > 0:
                x = torch.tensor(x, device=gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5

from concurrent.futures import ThreadPoolExecutor

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)

speed_test = True  # you have to enable this manually
    
def predict_on_demo_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join("input/combined", filename), batch_size=frames_per_video)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)


# -----------------

import random
import gc
import cv2
import glob
import copy
from PIL import Image
from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

package_path = 'input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(0)

# pretrained weights

# Set Trained Weight Path
weight_path = 'efficientnet_b0_epoch_15_loss_0.158.pth'
trained_weights_path = os.path.join('input/deepfake-detection-model-weight', weight_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True


def get_img_from_mov(video_file, num_img, frame_window):
    # https://note.nkmk.me/python-opencv-videocapture-file-camera/
    cap = cv2.VideoCapture(video_file)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_list = []
    for i in range(num_img):
        _, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * frame_window)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
            break
    cap.release()

    return image_list

class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img):
        return self.data_transform(img)
    

class DeepfakeDataset(Dataset):
    def __init__(self, file_list, device, detector, transform, img_num=20, frame_window=10):
        self.file_list = file_list
        self.device = device
        self.detector = detector
        self.transform = transform
        self.img_num = img_num
        self.frame_window = frame_window

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]
        img_list = []

        # Movie to Image
        try:
            all_image = get_img_from_mov(mov_path, self.img_num, self.frame_window)
        except:
            return [], mov_path.split('/')[-1]
        
        # Detect Faces
        for image in all_image:
            
            try:
                _image = image[np.newaxis, :, :, :]
                boxes, probs = self.detector.detect(_image, landmarks=False)
                x = int(boxes[0][0][0])
                y = int(boxes[0][0][1])
                z = int(boxes[0][0][2])
                w = int(boxes[0][0][3])
                image = image[y-15:w+15, x-15:z+15]
                
                # Preprocessing
                image = Image.fromarray(image)
                image = self.transform(image)
                
                img_list.append(image)

            except:
                img_list.append(None)
            
        # Padding None
        img_list = [c for c in img_list if c is not None]
        
        return img_list, mov_path.split('/')[-1]
    
# model
model2 = EfficientNet.from_name('efficientnet-b0')
model2._fc = nn.Linear(in_features=model2._fc.in_features, out_features=1)
model2.load_state_dict(torch.load(trained_weights_path, map_location=torch.device(device)))

# Prediction
def predict_dfdc(dataset, model):
    
    torch.cuda.empty_cache()
    pred_list = []
    path_list = []
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            pred = 0
            imgs, mov_path = dataset.__getitem__(i)
            
            # No get Image
            if len(imgs) == 0:
                pred_list.append(0.5)
                path_list.append(mov_path)
                continue
                
                
            for i in range(len(imgs)):
                img = imgs[i]
                
                output = model(img.unsqueeze(0).to(device))
                pred += torch.sigmoid(output).item() / len(imgs)
                
            pred_list.append(pred)
            path_list.append(mov_path)
            
    torch.cuda.empty_cache()
            
    return path_list, pred_list


# --------------------------------

import os.path

print("----Listening----")
while not os.path.exists('input/combined/test1.mp4'):
    time.sleep(1)

if os.path.isfile('input/combined/test1.mp4'):
    demo_videos = sorted([x for x in os.listdir("input/combined") if x[-4:] == ".mp4"])
    # inf-model
    predict_demo = predict_on_demo_set(demo_videos, num_workers=4)
    submission_df = pd.DataFrame({"filename": demo_videos, "label": predict_demo})
    submission_df.to_csv("output/submission-inf.csv", index=False)
    # eff-model
    # Config
    img_size = 120
    img_num = 15
    frame_window = 5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(img_size, mean, std)

    detector = MTCNN(image_size=img_size, margin=14, keep_all=False, factor=0.5, 
                     select_largest=False, post_process=False, device=device).eval()

    test_file = [os.path.join('input/combined', path) for path in os.listdir('input/combined') if path[-4:] == ".mp4"]

    dataset = DeepfakeDataset(test_file, device, detector, transform, img_num, frame_window)
    print(test_file, demo_videos)

    path_list, pred_list = predict_dfdc(dataset, model2)

    # Submission
    res = pd.DataFrame({
        'filename': path_list,
        'label': pred_list,
    })

    res.sort_values(by='filename', ascending=True, inplace=True)
    res.to_csv('output/submission-eff.csv', index=False)
    
    
else:
    raise ValueError("%s isn't a file!" % file_path)