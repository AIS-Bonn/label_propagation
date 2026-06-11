from transformers import ViTModel, AutoImageProcessor
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, mode="cls"):
        super().__init__()
        '''
        feature_type: cls | cls_pool | projected
        '''
        self.feature_name = f'ViT'
        self.processor = AutoImageProcessor.from_pretrained("google/vit-huge-patch14-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
        self.mode = mode

        # Pass dummy input through to get the feature shape:
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.model(dummy_input)

        self.feature_dim = output.last_hidden_state[:, 0].shape[-1]

    def __call__(self, images):
        return self.get_features(images)
    
    # def to(self, device):
    #     self.model = self.model.to(device)

    def get_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.mode == "cls":
            return outputs.last_hidden_state[:, 0]
        elif self.mode == "full":
            return outputs.last_hidden_state[:, 1:]

    def parameters(self):
        return self.model.parameters()
