import os
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.split('\t')[0] for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = (labels)

    def __getitem__(self, index):
        assert os.path.exists(os.path.join(self.img_path, self.img_filename[index]))
        img = read_image(os.path.join(self.img_path, self.img_filename[index]))   
        img = T.ToPILImage()(img)
        img = img.convert('RGB')
        img_name = self.img_filename[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index]
        return img, label, img_name

    def __len__(self):
        return len(self.img_filename)
