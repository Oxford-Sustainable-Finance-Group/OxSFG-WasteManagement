# OxSFG-WasteManagement

## Dependencies
* Python 3.7.3 (other versions may also work)
* Pytorch 1.4.0 (other versions may also work)
* tqdm
* numpy
* tensorboard_logger
* CUDA 10.1


If you want to this code for the classification of plant from different sector please update the following parameters in train file:

* --DATA_PATH -----> {images, train_idx, val_idx, test_idx}
   
* --IMG_DATA ----->  "this folder contain all images together (train, val and test)

* --log_environment ----->  save log fo training process

* --model_pth_path ----->  save trained model for training process

* --TRAIN_IMG_FILE ----->  "Ids of traing images"

* --TRAIN_LABEL_FILE -----> "true label traing images"

* --VAL_IMG_FILE ----->  Ids of val images

* --VAL_LABEL_FILE -----> true label val images

* --training_details ----->  default='/plant-classification/model/data_info.json',help="save all loss and accuracy"

* --batch_size----->  default=16, type=int, help="Total batch size for training
    
* --val_batch_size ----->  default=1, type=int, help="Total batch size for eval."
    
* --save_per_epoch ----->  default=1, help="Run prediction on validation set every so many steps. By default always run one evaluation at the end of training."

* --LR -----> default=1e-4, type=float, help="The initial learning rate ."
    
* --N_EPOCHS ----->  default=50, type=int, help="Total number of training epochs to perform."
    
* --num_classes -----> default=2, type=int, help="Total number classes in the dataset."




Soon I will update all the readme files as well.

Please bear with me!! :blush:
