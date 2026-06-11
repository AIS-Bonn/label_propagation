from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionModelWithProjection
import torch

MODELS_CLIP = {
        "base32": "openai/clip-vit-base-patch32",
        "base16": "openai/clip-vit-base-patch16",
        "large14": "openai/clip-vit-large-patch14",
        "large14-336": "openai/clip-vit-large-patch14-336"
}


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, model_type="base32", feature_type="projected"):
        super().__init__()
        '''
        feature_type: cls | cls_pool | projected
        '''
        self.model_type = model_type
        self.feature_type = feature_type
        self.feature_name = f'CLIP-{model_type}-{feature_type}'
        if self.model_type in MODELS_CLIP.keys():
            self.processor = CLIPProcessor.from_pretrained(MODELS_CLIP[self.model_type])
        else:
            raise NotImplementedError

        if self.feature_type == "projected":
            self.model = CLIPVisionModelWithProjection.from_pretrained(MODELS_CLIP[self.model_type])
        elif self.feature_type == "cls" \
            or self.feature_type == "cls_pool" or self.feature_type == 'full':
            self.model = CLIPVisionModel.from_pretrained(MODELS_CLIP[self.model_type])
        else:
            raise NotImplementedError

        # Pass dummy input through to get the feature shape:
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.model(dummy_input)

        if self.feature_type == "projected":
            self.feature_dim = output.image_embeds.shape[-1]
        elif self.feature_type == "cls" or self.feature_type == "full":
            self.feature_dim = output.last_hidden_state[:, 0].shape[-1]
        elif self.feature_type == "cls_pool":
            self.feature_dim = output.pooler_output.shape[-1]
        else:
            raise NotImplementedError

    def forward(self, images):
        return self.get_features(images, process=True)

    # def to(self, device, process=False):
    #     self.model = self.model.to(device)

    def get_features(self, images, process=True):
        '''
        Returns: CLIP representations, dependant on the feature_type:
                    cls: the cls token of the image
                    cls_pool: pooled cls representation
                    projected: pooled and projected representation (standard)
                 Shape: [1, feature_dim]
        '''
        if process:
            inputs = self.processor(images=images, return_tensors="pt")
        else:
            inputs = {"pixel_values": images}
        # if self.model.device == DEVICE:
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.feature_type == "projected":
            return outputs.image_embeds
        elif self.feature_type == "cls":
            return outputs.last_hidden_state[:, 0]
        elif self.feature_type == "cls_pool":
            return outputs.pooler_output
        elif self.feature_type == "full":
            return outputs.last_hidden_state[:, 1:]
        else:
            raise NotImplementedError

    def parameters(self):
        return self.model.parameters()
