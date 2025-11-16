import torch

import utils
from model_architecture import resnet, vgg, densenet, googlenet
import evaluation

def main():
    modelDir = "./checkpoint/googlenet.pth"

    #Parameters for the dataset
    batchSize = 64 
    
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


if __name__ == "__main__":
    main()