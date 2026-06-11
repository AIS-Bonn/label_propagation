import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fiftyone as fo
from tqdm import tqdm
from PIL import Image

from theia_tools.utils import read_image_bbox

# ==== Hopfield Classifier ====
class HopfieldClassifier(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes):
        super().__init__()
        self.project = nn.Linear(feature_dim, proj_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))
        self.proj_dim = proj_dim
        self.feature_name = "Hopfield"

    def forward(self, features):
        q = self.project(features)                                # (B, D)
        logits = q @ self.prototypes.T / self.proj_dim**0.5       # (B, C)
        attention = torch.softmax(logits, dim=-1)
        return logits, attention

    def predict(self, features):
        q = self.project(features)                                # (B, D)
        logits = q @ self.prototypes.T / self.proj_dim**0.5       # (B, C)
        attention = torch.softmax(logits, dim=-1).squeeze()
        ix = torch.argmax(attention, dim=-1)
        return ix, attention[ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model


class HopfieldEnsembleLayerNoRecombination(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.feature_name = f"HopfieldEnsemble:{self.num_heads}-headed"
        self.head_dim = proj_dim // num_heads

        # Layers:
        self.projections = nn.ParameterList([
            nn.Linear(feature_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.prototype_ensemble = nn.ParameterList([
            nn.Parameter(torch.randn(num_classes, self.head_dim))
            for _ in range(num_heads)
        ])


    def forward(self, features):
        attentions_list = []
        for head in range(self.num_heads):
            q = self.projections[head](features)
            prototypes = self.prototype_ensemble[head]
            attention  = torch.softmax(q @ prototypes.T / self.head_dim ** 0.5, dim=-1)
            values = attention @ prototypes

            attentions_list.append(attention)

        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        return None, attention

    def predict(self, features):
        attentions_list = []
        for head in range(self.num_heads):
            q = self.projections[head](features)
            prototypes = self.prototype_ensemble[head]
            logits = q @ prototypes.T / self.head_dim ** 0.5
            attention = torch.softmax(logits, dim=-1)

            attentions_list.append(attention)

        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        ix = torch.argmax(attention, dim=-1)
        return ix, attention[torch.arange(len(ix)), ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model


class HopfieldEnsembleLayer(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, num_heads):
        super().__init__()
        self.proj_dim = proj_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.feature_name = f"HopfieldEnsemble:{self.num_heads}-headed"
        self.head_dim = proj_dim // num_heads

        # Layers:
        self.projections = nn.ParameterList([
            nn.Linear(feature_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.prototype_ensemble = nn.ParameterList([
            nn.Parameter(torch.randn(num_classes, self.head_dim))
            for _ in range(num_heads)
        ])
        self.recombination = nn.Linear(self.proj_dim, self.feature_dim, bias=False)


    def forward(self, features):
        values_list, attentions_list = [], []
        for head in range(self.num_heads):
            q = self.projections[head](features)
            prototypes = self.prototype_ensemble[head]
            attention  = torch.softmax(q @ prototypes.T / self.head_dim ** 0.5, dim=-1)
            values = attention @ prototypes

            attentions_list.append(attention)
            values_list.append(values)

        recombined = self.recombination(torch.cat(values_list, dim=-1))
        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        return recombined, attention

    def predict(self, features):
        attentions_list = []
        for head in range(self.num_heads):
            q = self.projections[head](features)
            prototypes = self.prototype_ensemble[head]
            logits = q @ prototypes.T / self.head_dim ** 0.5
            attention = torch.softmax(logits, dim=-1)

            attentions_list.append(attention)

        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        ix = torch.argmax(attention, dim=-1)
        return ix, attention[torch.arange(len(ix)), ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model

class HopfieldEnsemble(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, num_heads, normalize_prototypes=True):
        super().__init__()
        self.proj_dim = proj_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.feature_name = f"HopfieldEnsemble:{self.num_heads}-headed"
        self.head_dim = proj_dim // num_heads

        # Layers:
        self.prototype_ensemble = nn.ParameterList([
            HopfieldBank(feature_dim, self.head_dim, num_classes, normalize_prototypes)
            for _ in range(num_heads)
        ])
        self.recombination = nn.Linear(self.proj_dim, self.feature_dim, bias=False)


    def forward(self, features, normalize_query=True):
        values_list, attentions_list = [], []
        for bank in self.prototype_ensemble:
            values, attention = bank(features, normalize_query)
            attentions_list.append(attention)
            values_list.append(values)

        recombined = self.recombination(torch.cat(values_list, dim=-1))
        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        return recombined, attention

    def predict(self, features, normalize_query=True):
        attentions_list = []
        for bank in self.prototype_ensemble:
            _, attention = bank(features, normalize_query)
            attentions_list.append(attention)

        attention = torch.stack(attentions_list, dim=0).mean(dim=0)
        ix = torch.argmax(attention, dim=-1)
        return ix, attention[torch.arange(len(ix)), ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model


class HopfieldBank(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, normalize_prototypes=True):
        super().__init__()
        self.proj_dim = proj_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.do_norm = normalize_prototypes

        # Layers:
        self.projection = nn.Linear(feature_dim, proj_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))

    def get_prototypes(self):
        return self._normalize_prototypes() if self.do_norm else self.prototypes

    def _normalize_prototypes(self):
        return self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)

    def forward(self, features, normalize_query=True):
        q = self.projection(features)
        if normalize_query:
            q = q / q.norm(dim=-1, keepdim=True)
        prototypes = self.get_prototypes()
        attention  = torch.softmax(q @ prototypes.T / self.proj_dim ** 0.5, dim=-1)
        value = attention @ prototypes
        return value, attention

    def predict(self, features, normalize_query=True):
        q = self.projection(features)
        if normalize_query:
            q = q / q.norm(dim=-1, keepdim=True)
        prototypes = self.get_prototypes()
        logits = q @ prototypes.T / self.proj_dim ** 0.5
        attention = torch.softmax(logits, dim=-1)
        ix = torch.argmax(attention, dim=-1)

        return ix, attention[ix]


class HopfieldEnsembleResidual(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, num_heads):
        super().__init__()
        self.ensemble_layer = HopfieldEnsembleLayer(feature_dim, proj_dim, num_classes, num_heads)
        self.classifier = nn.Linear(proj_dim, num_classes)
        self.feature_name = f"HopfieldEnsembleResidual"

    def forward(self, features):
        ensemble_features, attentions = self.ensemble_layer(features)

        residuals = features + ensemble_features
        logits = self.classifier(residuals)
        return logits, attentions

    def predict(self, features):
        logits, _ = self.forward(features)
        class_probs = F.softmax(logits, dim=-1)
        ix = torch.argmax(class_probs, dim=-1)
        return ix, class_probs[torch.arange(len(ix)), ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model

class HopfieldSequential(nn.Module):
    def __init__(self, feature_dim, proj_dim, num_classes, num_heads):
        super().__init__()
        self.layer1 = HopfieldEnsembleLayer(feature_dim, proj_dim, num_classes, num_heads)
        self.layer2 = HopfieldEnsembleLayer(proj_dim, proj_dim // 4, num_classes, num_heads * 2)
        self.feature_name = f"HopfieldSequential"

    def forward(self, features):
        l1_features, l1_votes = self.layer1(features)

        residuals = features + l1_features

        l2_features, l2_votes = self.layer2(l1_features)

        # final_votes = l1_votes / 3.0 + l2_votes * 2.0  / 3.0
        return l2_features, l2_votes

    def predict(self, features):
        _, final_votes  = self.forward(features)
        ix = torch.argmax(final_votes, dim=-1)
        return ix, final_votes[torch.arange(len(ix)), ix]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model


class HopfieldBoost(nn.Module):
    def __init__(self, models, learn_weights=False):
        super().__init__()
        self.models = models
        if learn_weights:
            self.weights = nn.Parameter(torch.randn(len(self.models), 1, 1))
        else:
            self.register_buffer('weights', torch.ones(len(self.models), 1, 1) / len(self.models))
        self.feature_name = "HopfieldBoost"

    def forward(self, features_list):
        attns = []
        with torch.no_grad():
            for model_ix, features in enumerate(features_list):
                _, attn = self.models[model_ix](features)
                attns.append(attn)
            attns = torch.stack(attns, dim=0)
        attns = (F.softmax(self.weights, dim=0) * attns).sum(0)
        return attns, attns

    def predict(self, features_list):
        _, final_votes  = self.forward(features_list)
        ix = torch.argmax(final_votes, dim=-1)
        return ix, final_votes[torch.arange(len(ix)), ix]

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model

def train_hopfield(model, feature_extractor, dataloader, torch_dataset, optimizer, device, loss_fn):
    model.train()
    losses = []
    pbar = tqdm(dataloader, "Training")
    for batch in pbar:
        images, labels = batch
        target_vectors = torch_dataset.convert_labels(labels)
        images, target_vectors = images.to(device), target_vectors.to(device)

        #  Assume CLIP encoder is frozen and loaded elsewhere
        with torch.no_grad():

            features = feature_extractor(images)

        loss = loss_fn(target_vectors, features, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        pbar.set_postfix(loss=avg_loss)
    return sum(losses) / len(dataloader)

def train_boost(model, extractors, dataloader, torch_dataset, optimizer, device, loss_fn):
    model.eval()
    losses = []
    pbar = tqdm(dataloader, "Training")
    for batch in pbar:
        images, labels = batch
        target_vectors = torch_dataset.convert_labels(labels)
        images, target_vectors = images.to(device), target_vectors.to(device)

        features_list = []
        with torch.no_grad():
            for feature_extractor in extractors:
                features = feature_extractor(images)
                features_list.append(features)

        loss = loss_fn(target_vectors, features_list, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        pbar.set_postfix(loss=avg_loss)
    return sum(losses) / len(dataloader)

def validate_fo(model, feature_extractor, fo_dataset, torch_dataset, device):
    model.eval()
    print(f"Labeling with {model.feature_name}")
    for sample in fo_dataset.iter_samples(progress="verbose"):
        detections = []
        if sample.ground_truth is not None:
            for gt_detection in sample.ground_truth.detections:
                # read the image ...
                img = read_image_bbox(sample.filepath, gt_detection.bounding_box)
                img = Image.fromarray(img)
                img = torch_dataset.pre_processor(img)
                img = torch.tensor(np.array(img)).unsqueeze(0)
                with torch.no_grad():
                    features = feature_extractor.get_features(img)
                    pred_ix, pred_attn = model.predict(features)
                pred_ix, pred_attn = int(pred_ix), float(pred_attn)
                pred_label = torch_dataset.get_label_names([pred_ix])[0]
                detections.append(
                        fo.Detection(
                            label=pred_label,
                            confidence=pred_attn,
                            bounding_box=gt_detection.bounding_box,
                            mask=gt_detection.mask
                        )
                )
        detections = fo.Detections(detections=detections)
        sample[model.feature_name] = detections
        sample.save()

    print("-" * 100)
    print(f"{model.feature_name} evaluation: ")
    eval_result = fo_dataset.evaluate_detections(
            model.feature_name,
            gt_field="ground_truth",
            use_masks=True,
            classes = None,
            compute_mAP=True
    )

    eval_result.print_report()
    map_score = eval_result.mAP()
    print(f"mAP score {map_score:.3f}")
    return eval_result

def validate_fo_boost(model, extractors, fo_dataset, torch_dataset, device):
    model.eval()
    print(f"Labeling with {model.feature_name}")
    for sample in fo_dataset.iter_samples(progress="verbose"):
        detections = []
        for gt_detection in sample.ground_truth.detections:
            # read the image ...
            img = read_image_bbox(sample.filepath, gt_detection.bounding_box)
            img = Image.fromarray(img)
            img = torch_dataset.pre_processor(img)
            img = torch.tensor(np.array(img)).unsqueeze(0)

            features_list = []
            with torch.no_grad():
                for feature_extractor in extractors:
                    features = feature_extractor(img)
                    features_list.append(features)
                pred_ix, pred_attn = model.predict(features_list)

            pred_ix, pred_attn = int(pred_ix), float(pred_attn)
            pred_label = torch_dataset.get_label_names([pred_ix])[0]
            detections.append(
                    fo.Detection(
                        label=pred_label,
                        confidence=pred_attn,
                        bounding_box=gt_detection.bounding_box,
                        mask=gt_detection.mask
                    )
            )
        detections = fo.Detections(detections=detections)
        sample[model.feature_name] = detections
        sample.save()

    print("-" * 100)
    print(f"{model.feature_name} evaluation: ")
    eval_result = fo_dataset.evaluate_detections(
            model.feature_name,
            gt_field="ground_truth",
            use_masks=True,
            classes = None,
            compute_mAP=True
    )

    eval_result.print_report()
    map_score = eval_result.mAP()
    print(f"mAP score {map_score:.3f}")
    return eval_result


def validate_hopfield(model, feature_extractor, dataloader, torch_dataset, device, loss_fn):
    model.eval()
    losses = []
    pbar = tqdm(dataloader, "Validation")

    with torch.no_grad():
        for batch in pbar:
            images, labels = batch
            target_vectors = torch_dataset.convert_labels(labels)
            images, target_vectors = images.to(device), target_vectors.to(device)

            #  Assume CLIP encoder is frozen and loaded elsewhere
            features = feature_extractor(images)

            loss = loss_fn(target_vectors, features, model)

            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            pbar.set_postfix(loss=avg_loss)
    return sum(losses) / len(dataloader)

def validate_boost(model, extractors, dataloader, torch_dataset, device, loss_fn):
    model.eval()
    losses = []
    pbar = tqdm(dataloader, "Validation")

    with torch.no_grad():
        for batch in pbar:
            images, labels = batch
            target_vectors = torch_dataset.convert_labels(labels)
            images, target_vectors = images.to(device), target_vectors.to(device)

            features_list = []
            with torch.no_grad():
                for feature_extractor in extractors:
                    features = feature_extractor(images)
                    features_list.append(features)

            loss = loss_fn(target_vectors, features_list, model)

            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            pbar.set_postfix(loss=avg_loss)
    return sum(losses) / len(dataloader)


def loss_cosine(target_vectors, features, model):
    _, attn = model(features)
    cos_sim = F.cosine_similarity(attn, target_vectors, dim=-1)
    loss = -cos_sim.mean()
    return loss

def loss_mse(target_vectors, features, model):
    _, attn = model(features)
    loss = F.mse_loss(attn, target_vectors)
    return loss

def loss_cross_entropy(target_vectors, features, model):
    logits, _ = model(features)
    loss = F.cross_entropy(logits, target_vectors)
    return loss

def regularization_intra(prototypes):
    prototypes_norm = F.normalize(prototypes, dim=-1)
    sim_matrix = prototypes_norm @ prototypes_norm.T
    off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    prototype_loss = sim_matrix[off_diag_mask].pow(2).mean()
    return prototype_loss

def regularization_inter(banks, class_ix):
    vi = torch.stack([bank[class_ix] for bank in banks], dim=0)
    vi_norm = F.normalize(vi, dim=-1)
    sims = vi_norm @ vi_norm.T
    off = sims[~torch.eye(sims.size(0), dtype=bool)]
    loss = off.pow(2).mean() if off.size(0) != 0 else torch.tensor(0., device=vi.device)
    return loss

def regularization_inter_cosine(proto_list):
    """
    proto_list: list of M prototype tensors, each (C, d_h)
    returns: scalar loss
    """
    C, _ = proto_list[0].shape
    loss = 0.0
    for i in range(C):
        # Stack the i-th prototypes across heads: (M, d_h)
        pi = torch.stack([P[i] for P in proto_list], dim=0)

        # Calculate cosine sim:
        pi_norm = F.normalize(pi, dim=-1)
        sims = 1 - pi_norm @ pi_norm.T

        # Calculate the loss:
        loss += torch.triu(sims).pow(2).mean()
    return loss / C

def regularization_fixpoint(proto_list):
    # Calculate the "clusters" the banks converge to:
    clusters = torch.stack([torch.mean(P, axis=0) for P in proto_list], axis=0)
    # Calculate the fixed point the banks should be tied to:
    fixed_point = clusters.mean(axis=0)

    # Calculate the distances:
    dists_to_fixpoint = ((clusters - fixed_point) ** 2).sum(dim=-1)
    dists_to_eachother = clusters.unsqueeze(1) - clusters.unsqueeze(0)
    dists_to_eachother = (dists_to_eachother ** 2).sum(dim=-1)

    # Calculate the loss given the above:
    loss_fixpoint = dists_to_fixpoint.mean() # Pull them towards the fixed point

    return loss_fixpoint

def regularization_rotation(proto_list):
    # Calculate the "clusters" the banks converge to:
    clusters = torch.stack([torch.mean(P, axis=0) for P in proto_list], axis=0)
    # Calculate the fixed point the banks should be tied to:
    fixed_point = clusters.mean(axis=0)

    # Calculate and normalize vectors from the fixed point:
    vecs_from_fixpoint = clusters - fixed_point
    vecs_from_fixpoint = F.normalize(vecs_from_fixpoint, dim=-1)

    # Calculate the cosine similarities between the vectors:
    sims = vecs_from_fixpoint @ vecs_from_fixpoint.T

    # Calculate the losss, we want to minimize the similarity between the vectors:
    loss_rotation = torch.triu(sims, diagonal=1).pow(2).mean()

    return loss_rotation


def regularization_inra_euclidean(prototypes):
    """
    prototypes: tensor of shape (C, d_h)
    returns: scalar loss
    """
    C, _ = prototypes.shape
    # Compute pairwise distances squared
    diffs = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)  # (C,C,d_h)
    dist_sq = (diffs ** 2).sum(dim=-1)                         # (C,C)
    # Zero out diagonal
    dist_sq = torch.triu(dist_sq)
    return dist_sq.mean()

def regularization_inter_euclidian(proto_list):
    """
    proto_list: list of M prototype tensors, each (C, d_h)
    returns: scalar loss
    """
    M = len(proto_list)
    C, _ = proto_list[0].shape
    loss = 0.0
    for i in range(C):
        # Stack the i-th prototypes across heads: (M, d_h)
        pi = torch.stack([P[i] for P in proto_list], dim=0)
        # pairwise diffs
        diffs = pi.unsqueeze(1) - pi.unsqueeze(0)   # (M,M,d_h)
        dist_sq = (diffs ** 2).sum(dim=-1)           # (M,M)
        loss += torch.triu(dist_sq).mean()
    return loss / C

def regularization_intra_attract(prototypes):
    """
    prototypes: tensor of shape (C, d_h)
    returns: scalar loss
    """
    P = F.normalize(prototypes, dim=-1)           # (C, d) on the unit sphere
    dots = P @ P.T                           # (C, C) matrix of p_i·p_j
    dots = torch.triu(dots, diagonal=1)
    # We want p_i·p_j → 1  ⇒  minimize -(p_i·p_j)
    return -dots.mean()

def regularization_inter_repel(proto_list):
    M = len(proto_list)
    loss = 0.0
    count = 0
    for m in range(M):
        for n in range(m+1, M):
            Pm = F.normalize(proto_list[m].get_prototypes(), dim=-1)
            Pn = F.normalize(proto_list[n].get_prototypes(), dim=-1)
            dots = (Pm * Pn).sum(dim=-1)     # (C,)
            loss += dots.mean()             # minimize p_m·p_n → -1
            count += 1
    return loss / count

def hopfield_loss(target_vectors, features, model, lambda_val):
    return loss_mse(target_vectors, features, model) + lambda_val * regularization_intra(model.prototypes)

def hopfield_ensemble_loss(target_vectors, features, model, lambda_intra, lambda_inter):
    intra_loss = [regularization_intra(prototypes) for prototypes in model.prototype_ensemble]
    intra_loss = torch.stack(intra_loss, dim=0).sum(dim=0)
    inter_loss = [regularization_inter(model.prototype_ensemble, class_ix) for class_ix in range(model.num_classes)]
    inter_loss = torch.stack(inter_loss, dim=0).sum(dim=0)
    return loss_cosine(target_vectors, features, model) + lambda_intra * intra_loss + lambda_inter * inter_loss


### The loss that produces the best results so far:
def hopfield_ensemble_loss_mse(target_vectors, features, model, lambda_intra, lambda_inter):
    loss = loss_mse(target_vectors, features, model)
    loss_intra = sum(regularization_inra_euclidean(Pm) for Pm in model.prototype_ensemble)
    loss_inter = regularization_inter_euclidian(model.prototype_ensemble)
    return loss + lambda_intra * loss_intra + lambda_inter * loss_inter
### ------------------------------------------------------------------------------------------

### Experimental loss function:
def hopfield_ensemble_loss_exp(target_vectors, features, model, lambda_intra, lambda_inter):
    loss = loss_mse(target_vectors, features, model)
    loss_intra = sum(regularization_intra_attract(bank.get_prototypes()) for bank in model.prototype_ensemble) / len(model.prototype_ensemble)
    loss_inter = regularization_inter_repel(model.prototype_ensemble)
    return loss + lambda_intra * loss_intra + lambda_inter * loss_inter
### ------------------------------------------------------------------------------------------

def hopfield_residual_loss(target_vectors, features, model, lambda_intra, lambda_inter):
    loss = loss_cross_entropy(target_vectors, features, model)
    loss_intra = sum(regularization_inra_euclidean(Pm) for Pm in model.ensemble_layer.prototype_ensemble)
    loss_inter = regularization_inter_euclidian(model.ensemble_layer.prototype_ensemble)
    return loss + lambda_intra * loss_intra + lambda_inter * loss_inter

def hopfield_sequential_loss_mse(target_vectors, features, model, lambda_intra, lambda_inter):
    loss = loss_mse(target_vectors, features, model)

    loss_intra1 = sum(regularization_inra_euclidean(Pm) for Pm in model.layer1.prototype_ensemble)
    loss_intra2 = sum(regularization_inra_euclidean(Pm) for Pm in model.layer2.prototype_ensemble)


    loss_inter1 = regularization_inter_euclidian(model.layer1.prototype_ensemble)
    loss_inter2 = regularization_inter_euclidian(model.layer2.prototype_ensemble)
    return loss + lambda_intra * (loss_intra1 + loss_intra2) + lambda_inter * (loss_inter1 + loss_inter2)

def hopfield_boost_loss(target_vectors, features_list, model):
    # we're only interested in weights which combine the models' votes:
    loss = loss_mse(target_vectors, features_list, model)
    return loss
