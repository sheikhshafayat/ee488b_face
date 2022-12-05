import torchvision

def MainModel(nOut=256, **kwargs):
    # return torchvision googlenet
    return torchvision.models.efficientnet_b3(num_classes=nOut)
