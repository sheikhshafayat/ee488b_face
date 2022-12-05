import torchvision

def MainModel(nOut=256, **kwargs):
    # return torchvision googlenet
    return torchvision.models.regnet_y_1_6gf(num_classes=nOut)
