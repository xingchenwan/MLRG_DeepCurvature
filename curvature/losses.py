import torch.nn.functional as F
import torch
import backpack


def cross_entropy(model, input, target, backpacked_model=False):
    """
    Evaluate the cross entropy loss.
    :param model:
    :param input:
    :param target:
    :param backpacked_model: if the model uses backpack facility, this toggle will backpack.extend() the
    loss function for the additional functionalities
    :return:
    """
    output = model(input)
    if backpacked_model:
        lossfunc = torch.nn.CrossEntropyLoss()
        lossfunc = backpack.extend(lossfunc)
        loss = lossfunc(output, target)
    else:
        loss = F.cross_entropy(output, target)
    return loss, output, {}


def cross_entropy_func(model, input, target):
    return lambda: model(input), lambda pred: F.cross_entropy(pred, target)