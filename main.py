import torch

import utils
from model_architecture import resnet, vgg, densenet, googlenet
import evaluation

import AttackWrappersWhiteBox

def main():
    modelDir = "./checkpoint/googlenet_v2.pth"

    #Parameters for the dataset
    batchSize = 64
    numClasses = 10
    
    #Define the GPU device we are using 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model (note this does not include pre-trained weights)
    # model = resnet.PreActResNet18().to(device)
    # model = vgg.VGG('VGG16').to(device)
    # model = densenet.DenseNet121().to(device)
    model = googlenet.GoogLeNet().to(device)

    #Load the trained weights
    checkpoint = torch.load(modelDir)
    
    # Remove 'module.' prefix from state_dict keys
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix (7 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)

    #Switch the model into eval model for testing
    model = model.eval()

    #Load in the dataset
    valLoader = utils.GetCIFAR10Validation(batchSize)
    
    # Evaluate model
    results = evaluation.evaluate_model(device, model, valLoader)
    
    # Print results
    evaluation.print_evaluation_results(results)

    # Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)

    #Check to make sure the accuracy is 100% on the correct loader
    correctAcc = utils.validateD(correctLoader, model, device)
    print("CIFAR-10 Clean Correct Loader Acc:", correctAcc)

    #Do the FGSM attack
    epsilonMax = 0.031 #Maximum perturbation
    clipMin = 0.0 #Minimum value a pixel can take
    clipMax = 1.0 #Maximum value a pixel can take 
    numSteps = 20
    epsilonStep = epsilonMax/numSteps

    #Run the attacks
    advLoader_FGSM = AttackWrappersWhiteBox.FGSMNativePytorch(device, correctLoader, model, epsilonMax, clipMin, clipMax)
    advLoader_PGD = AttackWrappersWhiteBox.PGDNativePytorch(device, correctLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax)
    
    # #Check the accuracy of the model on the adversarial examples 
    advAcc_FGSM = utils.validateD(advLoader_FGSM, model, device)
    print("CIFAR-10 FGSM Adversarial Acc:", advAcc_FGSM)
    
    advAcc_PGD = utils.validateD(advLoader_PGD, model, device)
    print("CIFAR-10 PGD Adversarial Acc:", advAcc_PGD)

    # Save Adversarial samples # make  totalSamplesRequired = 10 to save 1 sample from each class
    # xCleanTensor, yCleaTensor = utils.DataLoaderToTensor(correctLoader)
    # xAdvTensor, _ = utils.DataLoaderToTensor(advLoader_FGSM)
    # #DMP.ShowImages(xCleanTensor, xAdvTensor)
    # save_path = './sample_images/Output2.png'  # Adjust to the correct directory
    # utils.ShowImages(xCleanTensor, xAdvTensor, save_path=save_path)
    

if __name__ == "__main__":
    main()