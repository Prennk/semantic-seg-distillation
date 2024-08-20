import torch.nn as nn
from torchvision import models
import torchvision.models.segmentation as seg_model

class Create_DeepLabV3(nn.Module):
    """Create DeepLabV3 model with ResNet50 backbone from torchvision.
    keyword arguments:

    - num_classes (int): number of classes
    - args: argument from parser argument in main.py.
        - args.pretrained (str): pretrain model to load. choices: 'backbone' ResNet50 imagenet / 'all' COCO.
        - args.freeze (str): freeze the model
    - layers_to_hook (list(str)): list of layers name that will return feature maps
    """
    def __init__(self, num_classes, args, layers_to_hook=None):
        super(Create_DeepLabV3, self).__init__()
        if args.mode in ["train", "test"]:
            if args.pretrained == "backbone":
                print("Loading pretrained ResNet50 IMAGENET1K_V2...")
                weights_backbone = models.ResNet50_Weights.IMAGENET1K_V2
                weights = None
            elif args.pretrained == "all":
                print("Loading pretrained ResNet50 COCO_WITH_VOC_LABELS_V1...")
                weights_backbone = None
                weights = seg_model.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            elif not args.pretrained:
                weights_backbone = None
                weights = None
            else:
                raise ValueError(f"Unknown pretrained command: {args.pretrained}")
        elif args.mode == "distill":
            weights_backbone = None
            weights = None
        else:
            raise ValueError(f"Unknown argument {args.mode}")

        if args.model == "deeplabv3_resnet50":
            self.model = seg_model.deeplabv3_resnet50(
                    weights=weights, 
                    aux_loss=True,
                    weights_backbone=weights_backbone)
            self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        elif args.model == "deeplabv3_resnet101":
            self.model = seg_model.deeplabv3_resnet101(
                    weights=weights, 
                    aux_loss=True,
                    weights_backbone=weights_backbone)
            self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            raise ValueError(f"Unknown model {args.model}.")

        if args.mode in ["train", "test"]:
            if args.pretrained and args.freeze:
                # freeze deeplabv3 backbone
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