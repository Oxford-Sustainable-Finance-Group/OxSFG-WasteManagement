import numpy as np
import os
import sys
from tqdm import tqdm, trange
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np
import time
from torchvision.io import read_image
import torchvision.transforms as T
import argparse

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes = num_classes

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Softmax())

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filenames, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        self.img_filename = img_filenames
        
        
    def __getitem__(self, index):
        img = read_image(os.path.join(self.img_path, self.img_filename[index]))   
        assert os.path.exists(os.path.join(self.img_path, self.img_filename[index]))
        img = T.ToPILImage()(img)
        img = img.convert('RGB')
        img_name = self.img_filename[index]
        if self.transform is not None:
            img = self.transform(img)      
        return img, img_name

    def __len__(self):
        return len(self.img_filename)



def evaluate(test_loader, model, prediction_txt_path, prediction_plant_path, prediction_no_plant_path, device):
    result = {}
    plant = {}
    no_plant = {}
    for batch in tqdm(test_loader,desc="Testing"):#
        x, img_name = batch
        x, = x.to(device), 
        y_hat = model(x)
       #print((y_hat))
        for (output, img) in zip(y_hat, img_name):
            if (torch.argmax(output)==1):
                result[img] = [y_hat[0][0],y_hat[0][1], 'plant']
                plant[img] = [y_hat[0][0],y_hat[0][1]]#torch.max(y_hat)
            if (torch.argmax(output)==0):
                result[img] = [y_hat[0][0],y_hat[0][1], 'no plant']#torch.max(y_hat)
                no_plant[img] = [y_hat[0][0],y_hat[0][1]]#torch.max(y_hat)
    if os.path.exists(prediction_plant_path):
        a = 'a'
    else:
        a = 'w'
    with open(prediction_txt_path, a) as f:
        for img, s in result.items():
            img_id, lat, long = img.split('_')[0], img.split('_')[1], '.'.join([(img.split('_')[2]).split('.')[0],(img.split('_')[2]).split('.')[1]])
            f.write('%d\t%s\t%s\t%.5f\t%.5f\t%s\n' % (int(img_id), str(lat), str(long), s[0], s[1], s[2]))
    if os.path.exists(prediction_plant_path):
        a = 'a'
    else:
        a = 'w'
    with open(prediction_plant_path, a) as f:
        for img, s in plant.items():
            img_id, lat, long = img.split('_')[0], img.split('_')[1], '.'.join([(img.split('_')[2]).split('.')[0],(img.split('_')[2]).split('.')[1]])
            f.write('%d\t%s\t%s\t%.5f\t%.5f\n' % (int(img_id), str(lat), str(long), s[0], s[1]))
    if os.path.exists(prediction_no_plant_path):
        a = 'a'
    else:
        a = 'w'
    with open(prediction_no_plant_path, a) as f:
        for img, s in no_plant.items():
            img_id, lat, long = img.split('_')[0], img.split('_')[1], '.'.join([(img.split('_')[2]).split('.')[0],(img.split('_')[2]).split('.')[1]])
            f.write('%d\t%s\t%s\t%.5f\t%.5f\n' % (int(img_id), str(lat), str(long), s[0], s[1]))


def test(args, device):

    model = CustomConvNet(2)
    #model.load_state_dict(torch.load(model_pth_path))
    model.load_state_dict(torch.load(args.model_pth_path,map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    image_path = os.path.join(args.DATA_PATH, args.IMG_DATA)
    files_list = os.listdir(image_path)
    TEST_IMG_FILE = []    
    for img_file in files_list:
        if (img_file.endswith(".png")):
            TEST_IMG_FILE.append(img_file)

    test_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
    dset_test = DatasetProcessing(args.DATA_PATH, args.IMG_DATA, TEST_IMG_FILE, test_transforms)
    test_loader = DataLoader(dset_test,batch_size=1,shuffle=False,num_workers=1)
    evaluate(test_loader, model, args.prediction_txt_path,  args.prediction_plant_path,  args.prediction_no_plant_path, device)

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--DATA_PATH", default='/Users/aloksingh/Documents/Oxford/Waste-management/', help="this folder contain all data -----> {images, train_idx, val_idx, test_idx}")
   
    parser.add_argument("--IMG_DATA", default='Lithuania', help="this folder contain all images together (train, val and test).")

    parser.add_argument("--model_pth_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/cnn_model_model.pth',
     help="save trained model fo training process")

    parser.add_argument("--TEST_IMG_FILE", default='test-idx.txt',help="Ids of traing images")

    parser.add_argument("--TEST_LABEL_FILE", default='test-labels.txt',help="true label traing images")
    
    parser.add_argument("--test_batch_size", default=1, type=int, help="Total batch size for eval.")
        
    parser.add_argument("--num_classes", default=2, type=int, help="Total number classes in the dataset.")

    parser.add_argument("--prediction_txt_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/stage1-result/'+'stage1'+'_result.txt',
     help="save prediction generated by trained model on validation test set")

    parser.add_argument("--prediction_plant_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/stage1-result/'+'stage1'+'_plant_result.txt',
     help="this file contains coordinates having plant")

    parser.add_argument("--prediction_no_plant_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/stage1-result/'+'stage1'+'_no_plant_result.txt',
     help="this file contains coordinates not having plant")


    # Defining model and training options
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    
    # Training
    test(args, device)

if __name__ == '__main__':
    main()
