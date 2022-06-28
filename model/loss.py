from prometheus_client import Metric
import torch.nn.functional as F
import torch
import numpy as np
np.set_printoptions(suppress=True) # to not print scientific notation


def nll_loss(output, target):
    """ Negative log likelihood loss
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): target
    Returns:
        torch.Tensor: loss
    """
    # take only the target class which has a value of 1 to calculate the NLL loss
    target = torch.argmax(target, dim=1)
    return F.nll_loss(output, target)


def dice_loss(output, target, smooth_nr=1e-05, smooth_dr=1e-05):
    """ Dice loss
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
             https://docs.monai.io/en/stable/losses.html#diceloss
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): target
        smooth_nr (float): smooth factor, numerator
        smooth_nr (float): smooth factor, denominator
    Returns:
        torch.Tensor: loss
    """   
    # flatten label and prediction
    output = output.view(-1)
    target = target.view(-1)    

    intersection = (output * target).sum()                            
    dice_coeff = (2. * intersection + smooth_nr) / (output.sum() + target.sum() + smooth_dr)  

    return 1 - dice_coeff


# verify if this works! to deal with unbalanced classes problem
def focal_loss(output, target, alpha=0.1, gamma=3):
    """ Focal loss
        See: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Focal-Loss
             https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L08/code/cross-entropy-pytorch.ipynb
             and https://arxiv.org/pdf/1708.02002.pdf  
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): groundtruth target
        alpha (float): weight of the class with the highest probability
        gamma (float): focusing parameter, higher means more focusing on hard examples
    Returns:
        torch.Tensor: loss
    """  
    # Output and target both have shape (N, 4, H, W).
    # for nll_loss we require input=(N,C,H,W) and target=(N,H,W) so we take argmax for target
    target = torch.argmax(target, dim=1)
    CE = F.nll_loss(output, target)

    # compute cross-entropy and focal loss    
    CE_EXP = torch.exp(-CE)
    focal_loss = alpha * (1 - CE_EXP)**gamma * CE
                    
    return focal_loss


def generalized_wasserstein_loss(output, target):
    """ Generalized Wasserstein loss
        See: https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss
             https://arxiv.org/abs/1707.00478
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): groundtruth target
    Returns:
        torch.Tensor: loss
    """
    pass


# multi-class MSE loss
def mse_loss(output, target):
    """ Multi-class MSE loss
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): target
    Returns:
        torch.Tensor: loss
    """
    return F.mse_loss(output, target)

# categorical cross-entropy loss
def cce_loss(output, target):
    """ Categorical cross-entropy loss
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): target
    Returns:
        torch.Tensor: loss
    """
    return F.cross_entropy(output, target)