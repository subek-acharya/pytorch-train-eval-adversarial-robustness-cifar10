import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time

import utils
from model_architecture import resnet, vgg, densenet, googlenet

# Global variables
device = None
model = None
criterion = None
optimizer = None
trainloader = None
testloader = None
best_acc = 0

def train(epoch):
    global model, criterion, optimizer, trainloader, device
    
    print('\nEpoch: %d' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()       # Make all gradient values 0 at model parameter
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()      # Calculating gradients
        optimizer.step()     # Updating the weights

        train_loss += loss.item()   # total loss for this batch
        _, predicted = outputs.max(1)   # it returns two tensors value,index and we only need index
        total += targets.size(0)   # total data processes so far
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))       # for visualization of training

def test(epoch):
    global best_acc, model, criterion, testloader, device
    
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)  

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))  

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/googlenet.pth')
        best_acc = acc
        

def main():
    global device, model, criterion, optimizer, trainloader, testloader, best_acc

    # Define the GPU device we are using 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check available no of GPU/used for torch.nn.DataParallel
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Define Batch Size
    batchSize = 64
    
    # Defining Cifar10 classes
    CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch (last checkpoint not considered in this code)

    learning_rate = 0.01

    # Load CIFAR 10 data to dataloader
    
    # Creating the CIFAR10 training dataloader
    trainloader = utils.GetCIFAR10Training(batchSize)  
    # Creating the CIFAR10 testing dataloader
    testloader = utils.GetCIFAR10Validation(batchSize)  
        
    # Test the loaders
    print("==> Testing data loaders...")
    for images, labels in trainloader:
        print(f"Train batch shape: {images.shape}")
        print(f"Train labels shape: {labels.shape}")
        print(f"Train data range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    for images, labels in testloader:
        print(f"Test batch shape: {images.shape}")
        print(f"Test labels shape: {labels.shape}")
        print(f"Test data range: [{images.min():.3f}, {images.max():.3f}]")
        break

    # Define model
    # model = resnet.PreActResNet18().to(device)
    # model = vgg.VGG('VGG16').to(device)
    # model = densenet.DenseNet121().to(device)
    model = googlenet.GoogLeNet().to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # Defining Loss Function
    criterion = nn.CrossEntropyLoss()

    # Defining Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # scheduler makes learning rate smaller with time

    # Start tracking total training time
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    training_start_time = time.time()

    # Run training and testing and save the checkpoint
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Display comprehensive training summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Total epochs: 200")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()