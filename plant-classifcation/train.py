import sys
import os
import argparse
import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from tensorboard_logger import configure, log_value
from tqdm import tqdm, trange
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import dataset_processing
from torchvision.transforms import ToTensor
from tensorboard_logger import configure, log_value
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import json

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


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def evaluate(test_loader, model, criterion, device):
    correct, total = 0, 0
    test_loss = 0.0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for batch in (test_loader):
        x, y, img_name = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss.detach().cpu().item() / len(test_loader)
        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)

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
    recall = TP/(TP+FN) * 100
    precision = (TP/(FP+TP)) * 100
    f_score = 2*(recall*precision)/(recall+precision)
    return test_loss, correct / total * 100, recall, precision, f_score

def train(args, device):
    # Loading data
    #transform = ToTensor()
    configure(args.log_environment, flush_secs=10)

    train_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]) #used to make all input images of same type, dimension or to perform augmentation

    dset_train = dataset_processing.DatasetProcessing(args.DATA_PATH, args.IMG_DATA, args.TRAIN_IMG_FILE, args.TRAIN_LABEL_FILE, train_transforms)

    val_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

    dset_test = dataset_processing.DatasetProcessing(args.DATA_PATH, args.IMG_DATA, args.VAL_IMG_FILE, args.VAL_LABEL_FILE, val_transforms)


    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(dset_test, batch_size=args.val_batch_size, shuffle=True, num_workers=1)

    net = CustomConvNet(args.num_classes).to(device)
    N_EPOCHS = args.N_EPOCHS
    LR = args.LR
    total_step = len(train_loader)
    save_per_epoch = args.save_per_epoch
    saving_schedule = [int(x * total_step / save_per_epoch) for x in list(range(1, save_per_epoch + 1))]
    print('total: ', total_step)
    print('saving_schedule: ', saving_schedule)
    optimizer = Adam(net.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    val_aucc, train_accu, val_recall, val_precision, val_fs, train_ls, val_ls = [], [], [], [], [], [], []
    val_acc  = 0
    
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        log_value('epoch', epoch)
        correct = 0
        total = 0
        for i, batch in enumerate((train_loader)):#tqdm
            x, y, image_name = batch
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            log_value('train_loss', train_loss, epoch * total_step + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i+1 in saving_schedule:
                net.eval()
                loss, accuracy, recall, precision, f_score  = evaluate(test_loader, net, criterion, device)
                if accuracy>val_acc:
                    print("Model saving")
                    val_acc = accuracy
                    torch.save(net.state_dict(), args.model_pth_path)
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Val acc: {accuracy:.2f}")
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Val recall: {recall:.2f}")
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Val precision: {precision:.2f}")
                print(f"Epoch {epoch + 1}/{N_EPOCHS} Val f score: {f_score:.2f}")
                
                log_value('Learning rate', LR)
                val_aucc.append(accuracy)
                val_recall.append(recall)
                val_precision.append(precision)
                val_fs.append(f_score)

                log_value('val_aucc', accuracy, epoch)
                log_value('val_rec', recall, epoch)
                log_value('val_pre', precision, epoch)
                log_value('val_fsc', f_score, epoch)
                log_value('val_loss', loss, epoch)
        train_ls.append(train_loss)
        val_ls.append(loss)
        train_acc = correct / total * 100
        train_accu.append(train_acc)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} Train loss: {train_loss:.2f}\n")
        print(f"Epoch {epoch + 1}/{N_EPOCHS} Train acc: {train_acc:.2f}\n")
        log_value('train_acc', train_acc, epoch)
        
    data_dct = {'val_aucc':val_aucc,'val_rec':val_recall,'val_pre':val_precision,'val_fsc':val_fs,'val_loss':val_ls, 'train_loss':train_ls, 'train_acc':train_accu}
    
    with open(args.training_details, 'w') as fp:
        json.dump(data_dct, fp)



def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--DATA_PATH", default='data', help="this folder contain all data -----> {images, train_idx, val_idx, test_idx}")
   
    parser.add_argument("--IMG_DATA", default='All-images', help="this folder contain all images together (train, val and test).")

    parser.add_argument("--log_environment", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/log/',
     help="save log fo training process")

    parser.add_argument("--model_pth_path", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/cnn_model_model.pth',
     help="save trained model fo training process")

    parser.add_argument("--TRAIN_IMG_FILE", default='train_idx.txt',help="Ids of traing images")

    parser.add_argument("--TRAIN_LABEL_FILE", default='train-labels.txt',help="true label traing images")

    parser.add_argument("--VAL_IMG_FILE", default='validation-idx.txt',help="Ids of val images")

    parser.add_argument("--VAL_LABEL_FILE", default='validation-labels.txt',help="true label val images")

    parser.add_argument("--training_details", default='/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/data_info.json',help="save all loss and accuracy")

    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    
    parser.add_argument("--val_batch_size", default=1, type=int, help="Total batch size for eval.")
    
    parser.add_argument("--save_per_epoch", default=1, type=int, help="Run prediction on validation set every so many steps. Will always run one evaluation at the end of training.")

    parser.add_argument("--LR", default=1e-4, type=float, help="The initial learning rate .")
    
    parser.add_argument("--N_EPOCHS", default=50, type=int, help="Total number of training epochs to perform.")
    
    parser.add_argument("--num_classes", default=2, type=int, help="Total number classes in the dataset.")


    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    args = parser.parse_args()
    # Training
    train(args, device)

 
if __name__ == '__main__':
    main()
