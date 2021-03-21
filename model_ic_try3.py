import numpy as np
import time
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from datetime import datetime

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Credit for VGG variation: https://github.com/msyim/VGG16/blob/master/VGG16.py

# Define classifier class
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU())
    
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU())
    
    return layer

class Net(nn.Module):
    def __init__(self, n_classes=3):
        super(Net, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(8192, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        
        # past conv blocks, passing to fc layers
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        
        out = F.log_softmax(out, dim=1)
        
        return out

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define NN function
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data):
    # Create new model on CUDA device
    model = Net().to(device)
    history ={'train_loss':[],'val_loss':[],'val_accuracy':[]}
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    start = time.time()

    epochs = n_epoch
    steps = 0 
    running_loss = 0
    running_accuracy = 0
    print_every = 60
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            # All tensors need to(device) otherwise conflict
            # Convert back later when calculating with .extract().cpu()
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            
            running_ps = torch.exp(output)
            running_equality = (labels.data == running_ps.max(dim=1)[1])
            running_accuracy += running_equality.type(torch.FloatTensor).mean()
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
                    

                training_loss = running_loss/print_every
                validation_loss = test_loss/len(validloader)
                val_accuracy = accuracy/len(validloader)
                print(datetime.now().strftime("%Y-%m-%d, %H:%M:%S"), 
                      ": Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(training_loss),
                      "Test Loss: {:.3f} - ".format(validation_loss),
                      "Test Accuracy: {:.3f}".format(val_accuracy))
                history['train_loss'].append(training_loss)
                history['val_loss'].append(validation_loss)
                history['val_accuracy'].append(val_accuracy)

                running_loss = 0

                # Make sure training is back on
                model.train()
    
    
    print('epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model,history

# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    
def save_whole_model(model, path):
    torch.save(model, path)
    
# Define function to load model
def load_model(path):
    cp = torch.load(path)
    
    # Import pre-trained NN model 
    model = getattr(models, cp['model_name'])(pretrained=True)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False

    return model

def load_whole_model(path):
    model = torch.load(path)

    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False
      
    return model

def test_model(model, testloader, device='cuda'):  
    model.to(device)
    model.eval()
    accuracy = 0
    
    nb_classes = 3
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
#         print(images)
                
        output = model.forward(images)
        _, preds = torch.max(output, 1)
        ps = torch.exp(output)
#         print(ps.max(dim=1)[1])
#         print(labels.data)
        equality = (labels.data == ps.max(dim=1)[1])
#         print(equality)
        accuracy += equality.type(torch.FloatTensor).mean()
    
        for t,p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))
    print(confusion_matrix)
    
    # Print visualization of confusion matrix
    plt.figure(figsize=(8,8))

    class_names = ['normal','infected_covid','infected_non_covid']
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig("conf_matrix_small.png")
    
    return ps.max(dim=1)[1], labels.data