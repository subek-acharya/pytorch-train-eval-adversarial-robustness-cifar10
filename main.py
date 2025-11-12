import torch

import utils
from model_architecture import resnet, vgg, densenet, googlenet
import evaluation

def main():
    modelDir = "./checkpoint/resnet18.pth"

    #Parameters for the dataset
    batchSize = 64 
    
    #Define the GPU device we are using 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model (note this does not include pre-trained weights)
    model = resnet.PreActResNet18().to(device)
    # model = vgg.VGG('VGG16').to(device)
    # model = densenet.DenseNet121().to(device)
    # model = torch.nn.DataParallel(model)
    
    #Next load in the trained weights of the model 
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint['model'])

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