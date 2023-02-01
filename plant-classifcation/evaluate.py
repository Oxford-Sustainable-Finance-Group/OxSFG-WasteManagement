from torchvision import  transforms
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import dataset_processing
from torchvision.transforms import ToTensor
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
            nn.AdaptiveAvgPool2d((1, 1)))

def evaluate(args, test_loader, model, criterion, device):
    correct, total = 0, 0
    test_loss = 0.0
    result = {}
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    with torch.no_grad():
        for batch in (test_loader):#desc="Testing"
            
            x, y, img_name = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            for (output, img) in zip(y_hat, img_name):
                result[img] = (torch.argmax(output)).item()
            if (y ==1 and torch.argmax(y_hat, dim=1)==1):
                TP  +=1
            if (y ==1 and torch.argmax(y_hat, dim=1)==0 ):
                FN  +=1
            if (y ==0 and torch.argmax(y_hat, dim=1)==0 ):
                TN += 1
            if (y ==0 and torch.argmax(y_hat, dim=1)==1 ):
                FP +=1
     
    print(f"TP: {TP:.2f}")
    print(f"FN: {FN:.2f}")
    print(f"FP: {FP:.2f}")
    print(f"TN: {TN:.2f}")
    with open(args.prediction_txt_path, 'w') as f:
        for img, s in result.items():
            f.write('%s\t%s\n' % (str(img), str(s)))

    recall = TP/(TP+FN) * 100
    precision = (TP/(FP+TP)) * 100
    f_score = 2*(recall*precision)/(recall+precision)
    return test_loss, correct / total * 100, recall, precision, f_score

def test(args, device):
    model = CustomConvNet(args.num_classes)
    criterion = CrossEntropyLoss()
    model.load_state_dict(torch.load(args.model_pth_path,map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
    dset_test = dataset_processing.DatasetProcessing(args.DATA_PATH, args.IMG_DATA, args.TEST_IMG_FILE, args.TEST_LABEL_FILE, test_transforms)
    test_loader = DataLoader(dset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    [loss, accuracy, recall, precision, f_score] =  evaluate(args, test_loader, model, criterion, device)
    print("test loss: ",loss)
    print("Test accuracy:", accuracy)
    print("Test recall: ", recall)
    print("Test precision: ", precision)
    print("F score: ",f_score)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--DATA_PATH", default='data', help="this folder contain all data -----> {images, train_idx, val_idx, test_idx}")
   
    parser.add_argument("--IMG_DATA", default='All-images', help="this folder contain all images together (train, val and test).")

    parser.add_argument("--model_pth_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/cnn_model_model.pth',
     help="save trained model fo training process")

    parser.add_argument("--TEST_IMG_FILE", default='test-idx.txt',help="Ids of traing images")

    parser.add_argument("--TEST_LABEL_FILE", default='test-labels.txt',help="true label traing images")
    
    parser.add_argument("--test_batch_size", default=1, type=int, help="Total batch size for eval.")
        
    parser.add_argument("--num_classes", default=2, type=int, help="Total number classes in the dataset.")

    parser.add_argument("--prediction_txt_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/cnn_result.txt',
     help="save prediction generated by trained model on validation test set")


    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    args = parser.parse_args()
    # Training
    test(args, device)




if __name__ == '__main__':
    main()
