import torch.nn as nn
import torchvision.models.segmentation as seg_model

class Create_DeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=False, freeze=None, layers_to_hook=None):
        super(Create_DeepLabV3, self).__init__()
        weights = seg_model.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = seg_model.deeplabv3_resnet50(weights=weights, aux_loss=True, weights_backbone=None)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.feature_maps = {}
        self.layers_to_hook = layers_to_hook or []
        self._register_hooks()

        if pretrained and freeze:
            if freeze == "block":
                # freeze model except classifier and aux_classifier
                print(f"Freezing backbone...")
                print(f"Trainable: classifier blocks")
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                for param in self.model.aux_classifier.parameters():
                    param.requires_grad = True
            elif freeze == "layer":
                # freeze model except last classifier and aux_classifier layers
                print(f"Freezing backbone...")
                print(f"Trainable: last classifier layers")
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.classifier[4].parameters():
                    param.requires_grad = True
                for param in self.model.aux_classifier[4].parameters():
                    param.requires_grad = True                   

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