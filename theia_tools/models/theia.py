import os
import transformers
from transformers import AutoModel
import transformers.integrations.accelerate as _accel
import torch


MODELS_THEIA = {
    "base": "theaiinstitute/theia-base-patch16-224-cdiv",
    "small": "theaiinstitute/theia-small-patch16-224-cdiv",
    "tiny": "theaiinstitute/theia-tiny-patch16-224-cdiv"
}

# Override with THEIA_CACHE_DIR env var pointing to a local model directory
THEIA_LOCAL = os.environ.get(
    'THEIA_CACHE_DIR',
    os.path.join(os.path.expanduser('~'), '.cache', 'theia', 'theia-base-patch16-224-cdiv'),
)

def theia_features(theia_model, input_):
    '''
    Returns: the cls token of the theia representation, representation of the image
             [1, feature_dim]
    '''
    return theia_model.forward_feature(input_)[:, 0]

def vfm_features_fromtheia(theia_model, input_):
    return model(fake_input)


class TheiaFeatureExtractor(torch.nn.Module):
    def __init__(self, model_type="base", feature_type="theia"):
        super().__init__()
        self.model_type = model_type
        self.feature_name = f'Theia-{model_type}'
        self.feature_type = feature_type

        if model_type in MODELS_THEIA.keys():
            ### REMOVE REMOTE LOAD DUE TO SECTURITY VULNERABILITY:
            ### (torch.load() loads bin from DeIT backbone of Theia and is gonna complain)
            # self.model = AutoModel.from_pretrained(MODELS_THEIA[model_type], trust_remote_code=True, revision="08e8af3fbdc5912329c6cf411e7fe7df2f22686f")
            self.model = AutoModel.from_pretrained(THEIA_LOCAL, trust_remote_code=True, local_files_only=True)
        else:
            raise NotImplementedError

        # Pass dummy input through to get the feature shape:
        dummy_input = torch.ones(1, 3, 224, 224)
        with torch.no_grad():
            if self.feature_type == "theia" or self.feature_type == 'full':
                output = self.model.forward_feature(dummy_input).mean(axis=1)
            elif self.feature_type == 'dinov2':
                output = self.model.forward(dummy_input)['facebook/dinov2-large'].mean(axis=1)
            else:
                raise NotImplementedError
        self.feature_dim = output.shape[-1]

    
    def forward(self, images):
        return self.get_features(images)

    # def to(self, device):
    #     self.model = self.model.to(device)

    def get_features(self, images):
        inputs = images.to(self.model.device)
        with torch.no_grad():
            if self.feature_type == "theia":
                output = self.model.forward_feature(inputs).mean(axis=1)
            elif self.feature_type == 'dinov2':
                output = self.model.forward(inputs)['facebook/dinov2-large'].mean(axis=1)
            elif self.feature_type == 'full':
                output = self.model.forward_feature(inputs)
            else:
                raise NotImplementedError
        return output

    def parameters(self):
        return self.model.parameters()
