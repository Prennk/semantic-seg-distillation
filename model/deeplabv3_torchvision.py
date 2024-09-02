import torch.nn as nn
from torchvision import models
import torchvision.models.segmentation as seg_model

class Create_DeepLabV3_ResNet101(nn.Module):
    def __init__(self, num_classes, args, layers_to_hook=None):
        super(Create_DeepLabV3_ResNet101, self).__init__()
        print("Preparing model: DeepLabV3-ResNet101...")
        if args.mode in ["train", "test", "distill"]:
            if args.pretrained and args.mode != "distill":
                print("Loading pretrained ResNet101_Weights.IMAGENET1K_V2...")
                weights_backbone = models.ResNet101_Weights.IMAGENET1K_V2
            elif not args.pretrained or args.mode == "distill":
                weights_backbone = None
            else:
                raise ValueError(f"Unknown pretrained command: {args.pretrained}")
        else:
            raise ValueError(f"Unknown argument {args.mode}")
        
        self.model = seg_model.deeplabv3_resnet101(
                weights=None, 
                aux_loss=True,
                weights_backbone=weights_backbone)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

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
        available_layers = [name for name, _ in self.model.named_modules()]
        invalid_layers = [layer for layer in self.layers_to_hook if layer not in available_layers]
        if invalid_layers:
            raise NameError(f"The following layers are not found in the model: {invalid_layers}")

        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                module.register_forward_hook(self._hook(name))
    
    def _hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_maps[layer_name] = output.detach()

        return hook_fn

    def get_feature_map(self, layer_name):
        return self.feature_maps.get(layer_name, None)
    

class Create_DeepLabV3_ResNet18(nn.Module):
    def __init__(self, num_classes, args, layers_to_hook=None):
        super(Create_DeepLabV3_ResNet18, self).__init__()
        print("Preparing model: DeepLabV3-ResNet18...")
        if args.mode in ["train", "test", "distill"]:
            if args.pretrained and args.mode != "distill":
                print("Loading pretrained ResNet18_Weights.IMAGENET1K_V1...")
                weights_backbone = models.ResNet18_Weights.IMAGENET1K_V1
            elif not args.pretrained or args.mode == "distill":
                weights_backbone = None
            else:
                raise ValueError(f"Unknown pretrained command: {args.pretrained}")
        else:
            raise ValueError(f"Unknown argument {args.mode}")
        
        self.model = seg_model.deeplabv3_resnet18(
                weights=None, 
                aux_loss=True,
                weights_backbone=weights_backbone)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

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
        available_layers = [name for name, _ in self.model.named_modules()]
        invalid_layers = [layer for layer in self.layers_to_hook if layer not in available_layers]
        if invalid_layers:
            raise NameError(f"The following layers are not found in the model: {invalid_layers}")

        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                module.register_forward_hook(self._hook(name))
    
    def _hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_maps[layer_name] = output.detach()

        return hook_fn

    def get_feature_map(self, layer_name):
        return self.feature_maps.get(layer_name, None)