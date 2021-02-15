"""
Illustration of the drop-in replacement aspect of MomentumNets.

- Use example/from_resnet_to_momentumnet.py to transform ResNets into MomentumNets, pretrained or not.

"""

from torchvision.models import resnet18, resnet34, resnet101, resnet152
from momentumnet.models import BasicBlock, MBasicBlock, Bottleneck, MBottleneck, MResNet


def transform(model, pretrained=False, gamma=0.9, mem=False):
    """Return the MomentumNet counterpart of the model


        Parameters
        ----------
        model : a torchvision model
            The resnet one desires to turn into a momentumnet
        pretrained : bool (default: False)
            Whether using a pretrained resnet and transfer its weights
        gamma : float (default: 0.9)
            The momentum term for the MomentumNet
        mem : bool (default: False)
            If True then the MomentumNet has a smaller memory footprint

        Return
        ------
        mresnet : the MomentumNet counterpart of model

        """
    resnet = model
    layers = [len(resnet.layer1), len(resnet.layer2), len(resnet.layer3), len(resnet.layer4)]
    num_classes = resnet.fc.out_features
    try:
        _ = resnet.layer1[0].conv3
        mresnet = MResNet(Bottleneck, MBottleneck, layers, num_classes, gamma=gamma, mem=mem)
    except:
        mresnet = MResNet(BasicBlock, MBasicBlock, layers, num_classes, gamma=gamma, mem=mem)
    params1 = resnet.named_parameters()
    params2 = mresnet.named_parameters()
    if pretrained:
        for (name1, param1), (name2, param2) in zip(params1, params2):
            param2.data.copy_(param1)
    return mresnet

if __name__ == '__main__':
    resnet18 = resnet18(pretrained=True)
    mresnet18 = transform(resnet18, pretrained=True, mem=True)
    resnet34 = resnet34(pretrained=True)
    mresnet34 = transform(resnet34, pretrained=True)
    resnet101 = resnet101()
    mresnet101 = transform(resnet101, gamma=0.99, mem=True)
    resnet152 = resnet152()
    mresnet152 = transform(resnet152, gamma=0.99)
    print('Successfully transformed the 4 models')

