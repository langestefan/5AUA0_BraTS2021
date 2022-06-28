import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from monai.metrics import compute_generalized_dice, compute_meandice

# Basic accuracy
def accuracy(output, target):
    """Computes the accuracy of model output with respect to target.
    Args:
        output (torch.Tensor): The model output.
        target (torch.Tensor): The segmentation map.

    Returns:
        float: The accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

# Top-K accuracy
def top_k_acc(output, target, k=3):
    """Computes the top-k accuracy of model output with respect to target.
    Args:
        output (torch.Tensor): The model output.
        target (torch.Tensor): The segmentation map.
        k (int): Nr of samples to consider.

    Returns:
        float: The top-k accuracy.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# Sensitivity 
def sensitivity(output, target):
    """Computes the sensitivity of model output with respect to target.
    Args:
        output (torch.Tensor): The model output.
        target (torch.Tensor): The segmentation map.

    Returns:
        float: The sensitivity.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TP = 0
        FN = 0
        for i in range(len(target)):
            if pred[i] == target[i]:
                TP += 1
            else:
                FN += 1
    return TP / (TP + FN)

# Specificity
def specificity(output, target):
    """Computes the specificity of model output with respect to target.
    Args:
        output (torch.Tensor): The model output.
        target (torch.Tensor): The segmentation map.

    Returns:
        float: The specificity.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TN = 0
        FP = 0
        for i in range(len(target)):
            if pred[i] != target[i]:
                TN += 1
            else:
                FP += 1
    return TN / (TN + FP)


def dice_coeff_enhancing_tumor(output, target):
    """ Dice coeffecient for enhancing tumor(class 3)
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
    Args:
        output (torch.Tensor): model output probalities between [0-1], shape (N, C, H, W)
        target (torch.Tensor): target one-hot encoded, shape (N, C, H, W)
        eps (int): prevent division by 0
    Returns:
        torch.Tensor: dice coeff. Shape: (1,)
    """
    with torch.no_grad():

        # we only look at 3 (enhancing tumor), we set edema and necrosis to 0 (background)
        target_union = torch.sum(target[:, [3]], dim=1).clip(0, 1).unsqueeze(0).unsqueeze(0)

        # for output, we first take argmax over all classes. Then, we take the union of classes 1 and 3.
        output_argmax = torch.argmax(output, dim=1) # (N, H, W)
        
        # set edema (2) and necrosis (1) to background (0) 
        output_argmax[output_argmax == 1] = 0
        output_argmax[output_argmax == 2] = 0
        output_union = output_argmax.clip(0, 1).unsqueeze(0).unsqueeze(0)
        
        # compute dice coeff
        dice_coeff_et = dice_coeff(output_union, target_union)

    return dice_coeff_et


def dice_coeff_tumor_core(output, target):
    """ Dice coeffecient for tumor core (union of classes 1+3)
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
    Args:
        output (torch.Tensor): model output probalities between [0-1], shape (N, C, H, W)
        target (torch.Tensor): target one-hot encoded, shape (N, C, H, W)
        eps (int): prevent division by 0
    Returns:
        torch.Tensor: dice coeff. Shape: (1,)
    """
    with torch.no_grad():

        # for tumor core we only care about classes 1 (necrosis) and 3 (enhancing tumor), we set edema to 0 (background)
        target_union = torch.sum(target[:, [1, 3]], dim=1).clip(0, 1).unsqueeze(0).unsqueeze(0)

        # for output, we first take argmax over all classes, then we take the union of classes 1 and 3.
        output_argmax = torch.argmax(output, dim=1)
        
        # set edema (2) to background (0) so we are left with classes 0, 1 and 3 
        output_argmax[output_argmax == 2] = 0
        output_union = output_argmax.clip(0, 1).unsqueeze(0).unsqueeze(0)
        
        # compute dice coeff
        dice_coeff_tc = dice_coeff(output_union, target_union)

    return dice_coeff_tc



def dice_coeff_whole_tumor(output, target):
    """ Dice coeffecient for whole tumor (union of classes 1-3)
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
    Args:
        output (torch.Tensor): model output probalities between [0-1], shape (N, C, H, W)
        target (torch.Tensor): target one-hot encoded, shape (N, C, H, W)
        eps (int): prevent division by 0
    Returns:
        torch.Tensor: dice coeff. Shape: (1,)
    """
    with torch.no_grad():

        # for the target we want to take the union of all the tumor classes
        target_union = torch.sum(target[:, 1:], dim=1).clip(0, 1).unsqueeze(0).unsqueeze(0)

        # for the output we want to take the argmax of all the classes
        output_union = torch.argmax(output, dim=1).clip(0, 1).unsqueeze(0).unsqueeze(0)

        # compute dice coeff
        dice_coeff_wt = dice_coeff(output_union, target_union)

    return dice_coeff_wt



def dice_coeff(output, target):
    """ Dice coeffecient
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss  

    Args:
        output (torch.Tensor): model output binary encoded prediction [0, 1], shape (N, H, W)
        target (torch.Tensor): target binary encoded [0, 1], shape (N, H, W)

    Returns:
        torch.Tensor: dice coeff. Shape: (1,)
    """
    with torch.no_grad():

        # compute mean dice coeff over 3D volume
        dice_score = torch.as_tensor(compute_meandice(output, target, ignore_empty=False))
        dice_score = torch.mean(dice_score)

    return dice_score  
