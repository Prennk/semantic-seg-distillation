import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead, DeepLabV3, IntermediateLayerGetter

class ModifiedResNet101(nn.Module):
    def __init__(self, weights_backbone,):
        super(ModifiedResNet101, self).__init__()
        deeplabv3 = deeplabv3_resnet101(weights=None, weights_backbone=weights_backbone, aux_loss=True)
        resnet = deeplabv3.backbone
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ModifiedDeepLabV3(nn.Module):
    def __init__(self, num_classes, weights_backbone):
        super(ModifiedDeepLabV3, self).__init__()
        self.backbone = IntermediateLayerGetter(
            ModifiedResNet101(weights_backbone), return_layers={'layer3': 'aux', 'layer4': 'out'}
        )
        self.classifier = DeepLabHead(2048, num_classes)
        self.aux_classifier = FCNHead(1024, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier[0](features['out'])
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        aux = self.aux_classifier(features['aux'])
        aux = nn.functional.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)

        return x, aux

class Create_DeepLabV3(nn.Module):
    def __init__(self, num_classes, args, layers_to_hook=None):
        super(Create_DeepLabV3, self).__init__()
        if args.mode in ["train", "test"]:
            if args.pretrained:
                print("Loading pretrained models.ResNet101_Weights.IMAGENET1K_V2...")
                weights_backbone = models.ResNet101_Weights.IMAGENET1K_V2
            elif not args.pretrained or args.mode == "distill":
                weights_backbone = None
            else:
                raise ValueError(f"Unknown pretrained command: {args.pretrained}")
        else:
            raise ValueError(f"Unknown argument {args.mode}")
        
        self.model = ModifiedDeepLabV3(num_classes=num_classes, weights_backbone=weights_backbone)

        if args.mode in ["train", "test"]:
            if args.pretrained and args.freeze:
                print(f"Freezing backbone...")
                print(f"Trainable: DeepLabV3 head => ASPP + classifier")
                self.model.aux_classifier = None
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            elif args.pretrained and not args.freeze:
                print("[Warning] Pretrained layers is trainable")
            elif not args.pretrained and args.freeze:
                raise ValueError("You freeze the untrain model backbone")
        elif args.mode == "distill":
            print("Preparing DeepLabV3 for teacher...")
        else:
            raise ValueError(f"Unknown argument {args.mode}")
                
        self.feature_maps = {}
        self.layers_to_hook = layers_to_hook or []
        self._register_hooks() 

    def forward(self, x):
        output = self.model(x)

        return output

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                module.register_forward_hook(self._hook(name))
    
    def _hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_maps[layer_name] = output.detach()

        return hook_fn

    def get_feature_map(self, layer_name):
        return self.feature_maps.get(layer_name, None)