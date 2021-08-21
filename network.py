import torch
import torchvision


class full_network(torch.nn.Module):
    def __init__(self, backbone):
        super(full_network, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)

        return x


def initizalize_network(architecture, no_classes, load_model=None, pretrained=True):
    backbone = initialize_backbone(architecture=architecture, no_classes=no_classes, 
                                   pretrained=pretrained)
    if load_model == 'None':
        load_model = None
    if load_model is not None:
        state = torch.load(load_model, map_location='cpu')
        backbone_state_dict = {(key.split('.', 1)[1] if key.startswith('backbone.') else key):
                               value for key, value in state['state_dict'].items()}
        try:
            backbone.load_state_dict(backbone_state_dict)
        except RuntimeError:
            model_dict = backbone.state_dict()
            backbone_state_dict = {key: value for key, value in backbone_state_dict.items() 
                                   if ('fc' not in key and 'classifier' not in key)}
            model_dict.update(backbone_state_dict) 
            backbone.load_state_dict(model_dict)
            not_loaded = set(model_dict.keys()) - set(backbone_state_dict.keys())
            for keys in not_loaded:
                print('Not loaded in the backbone:', keys)

    return full_network(backbone=backbone)


def initialize_backbone(architecture, no_classes, pretrained):
    if architecture.lower() == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(512, no_classes)
    elif architecture.lower() == 'resnet50':
        net = torchvision.models.resnet50(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(2048, no_classes)
    elif architecture.lower() == 'resnet101':
        net = torchvision.models.resnet101(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(2048, no_classes)

    return net
