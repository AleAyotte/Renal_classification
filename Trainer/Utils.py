def init_weights(m):
    """
    Initialize the weights of the fully connected layer and convolutional layer with Xavier normal initialization
    and Kamming normal initialization respectively.

    :param m: A torch.nn module of the current model. If this module is a layer, then we initialize its weights.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)

    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)

    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.ones_(m.weight)
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)

def to_one_hot(inp, num_classes, device="cuda:0"):
    """
    Transform a logit ground truth to a one hot vector
    :param inp: The input vector to transform as a one hot vector
    :param num_classes: The number of classes in the dataset
    :param device: The device on which the result will be return. (Default="cuda:0", first GPU)
    
    :return: A one hot vector that represent the ground truth
    """
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return torch.autograd.Variable(y_onehot.to(device), requires_grad=False)