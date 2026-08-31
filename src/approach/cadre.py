"""CADRE for exemplar-free class-incremental learning.

CADRE couples adaptive response distillation and structure-preserving feature
distillation with multi-prototype classifier calibration. The implementation
stores network parameters and class-conditional statistics, but no images or
individual features from previous tasks.
"""

import copy
import torch
import numpy as np
from argparse import ArgumentParser
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Calibrated Adaptive Distillation with Relational Embedding (CADRE).

    The full method combines adaptive response distillation, relational feature
    distillation, and multi-prototype classifier calibration.
    """

    def __init__(self, model, device, nepochs=300, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=50,
                 clipgrad=10000, momentum=0.9, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1,
                 fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None,
                 balanced_bs=128, balanced_epochs=10, balanced_lr=0.01, use_rotation=False,
                 lambda_distill=2.0, distill_temperature=2.0, lambda_feat_distill=1.0,
                 adaptive_temp=2.0, lambda_relation=1.0,
                 num_subprototypes=3, min_samples_for_multimodal=10, covariance_epsilon=1e-4,
                 capture_analysis=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                   momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                                   eval_on_train, logger, exemplars_dataset)

        self.balanced_bs = balanced_bs
        self.balanced_epochs = balanced_epochs
        self.balanced_lr = balanced_lr
        self.use_rotation = use_rotation

        self.lambda_distill = lambda_distill
        self.distill_temperature = distill_temperature
        self.lambda_feat_distill = lambda_feat_distill
        self.adaptive_temp = adaptive_temp
        self.lambda_relation = lambda_relation
        self.num_subprototypes = num_subprototypes
        self.min_samples_for_multimodal = min_samples_for_multimodal
        self.covariance_epsilon = covariance_epsilon

        self.capture_analysis = capture_analysis
        self.analysis_pre_calibration_state = None
        self.analysis_pre_calibration_task = None

        self.old_model = None
        self.auxiliary_classifier = None
        self.prototypes = []
        self.prototype_labels = []
        self.gaussians = {}
        self.class_stats = {}
        self.feature_size = None

        print('\n' + '=' * 108)
        print('CADRE hyperparameters:')
        print(f'  lambda_distill={self.lambda_distill}  distill_temperature={self.distill_temperature}  '
              f'lambda_feat_distill={self.lambda_feat_distill}')
        print(f'  adaptive_temp={self.adaptive_temp}  lambda_relation={self.lambda_relation}')
        print(f'  num_subprototypes={self.num_subprototypes}  '
              f'min_samples_for_multimodal={self.min_samples_for_multimodal}  '
              f'covariance_epsilon={self.covariance_epsilon}')
        print(f'  balanced_bs={self.balanced_bs}  balanced_epochs={self.balanced_epochs}  '
              f'balanced_lr={self.balanced_lr}  use_rotation={self.use_rotation}')
        print('=' * 108 + '\n')

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()

        parser.add_argument('--balanced-bs', default=128, type=int, required=False)
        parser.add_argument('--balanced-epochs', default=10, type=int, required=False)
        parser.add_argument('--balanced-lr', default=0.01, type=float, required=False)
        parser.add_argument('--use-rotation', action='store_true')

        parser.add_argument('--lambda-distill', default=3.0, type=float, required=False)
        parser.add_argument('--distill-temperature', default=1.0, type=float, required=False)
        parser.add_argument('--lambda-feat-distill', default=1.0, type=float, required=False)
        parser.add_argument('--adaptive-temp', default=2.0, type=float, required=False)
        parser.add_argument('--lambda-relation', default=1.0, type=float, required=False)
        parser.add_argument('--num-subprototypes', default=3, type=int, required=False)
        parser.add_argument('--min-samples-for-multimodal', default=10, type=int, required=False)
        parser.add_argument('--covariance-epsilon', default=1e-4, type=float, required=False)

        parser.add_argument('--capture-analysis', action='store_true',
                          help='Capture the pre-calibration model state for offline mechanism analysis')

        return parser.parse_known_args(args)

    def _get_optimizer(self):
        if self.auxiliary_classifier is not None:
            params = list(self.model.parameters()) + list(self.auxiliary_classifier.parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum, foreach=False
        )

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0 and self.use_rotation:
            with torch.no_grad():
                sample_imgs, _ = next(iter(trn_loader))
                sample_imgs = sample_imgs[:1].to(self.device)
                sample_feats = self.model(sample_imgs, return_features=True)[1]
                self.feature_size = sample_feats.shape[1]

            num_classes = len(np.unique(trn_loader.dataset.labels))
            self.auxiliary_classifier = nn.Linear(self.feature_size, num_classes * 4)
            self.auxiliary_classifier.to(self.device)
            print(f'Initialized auxiliary classifier for rotation with {num_classes * 4} outputs')
        else:
            self.auxiliary_classifier = None
            if t > 0:
                print('Freezing Batch Normalization layers')
                self.model.freeze_bn()

        if t > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False

        if self.feature_size is None:
            with torch.no_grad():
                sample_imgs, _ = next(iter(trn_loader))
                sample_imgs = sample_imgs[:1].to(self.device)
                sample_feats = self.model(sample_imgs, return_features=True)[1]
                self.feature_size = sample_feats.shape[1]

        super(Appr, self).train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if t > 0:
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        if self.auxiliary_classifier is not None:
            self.auxiliary_classifier.train()

        for images, targets in tqdm(trn_loader, desc=f'Task {t}', leave=False):
            images, targets = images.to(self.device), targets.to(self.device)

            if t == 0 and self.use_rotation:
                images, targets = self._apply_rotation(images, targets, t)

            if t > 0 and self.old_model is not None:
                with torch.no_grad():
                    old_outputs = self.old_model(images)
                    old_features = self.old_model(images, return_features=True)[1]
            else:
                old_outputs = None
                old_features = None

            outputs = self.model(images)
            features = self.model(images, return_features=True)[1]

            if t == 0 and self.use_rotation:
                rot_outputs = self.auxiliary_classifier(features)
                outputs[t] = torch.cat([outputs[t], rot_outputs], dim=1)

            loss = self.criterion(t, outputs, targets, features, old_features, old_outputs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets, features, old_features=None, old_outputs=None):
        adjusted_targets = targets - self.model.task_offset[t]
        cls_loss = F.cross_entropy(outputs[t], adjusted_targets, label_smoothing=0.1)
        total_loss = cls_loss

        if t > 0:
            if old_outputs is not None and self.lambda_distill > 0:
                logit_distill_loss = self._compute_logit_distillation(
                    outputs, old_outputs, t, temperature=self.distill_temperature
                )
                total_loss += self.lambda_distill * logit_distill_loss

            if old_features is not None and self.lambda_feat_distill > 0:
                feat_distill_loss = self._compute_feature_distillation(features, old_features)
                total_loss += self.lambda_feat_distill * feat_distill_loss

        return total_loss

    def _compute_logit_distillation(self, new_outputs, old_outputs, task_id, temperature=2.0):
        """Weight task-wise KL terms by the current teacher-student discrepancy."""
        if task_id == 0:
            return torch.tensor(0.0, device=new_outputs[0].device)

        task_kls = []
        for old_task_id in range(task_id):
            if old_task_id < len(old_outputs) and old_task_id < len(new_outputs):
                new_logits = new_outputs[old_task_id]
                old_logits = old_outputs[old_task_id].detach()
                kl_loss = F.kl_div(
                    F.log_softmax(new_logits / temperature, dim=1),
                    F.softmax(old_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                task_kls.append(kl_loss)

        if len(task_kls) == 0:
            return torch.tensor(0.0, device=new_outputs[0].device)

        task_kls = torch.stack(task_kls)

        # Adaptive weighting: weights from detached KL so gradient cannot game them
        with torch.no_grad():
            num_old_tasks = task_kls.shape[0]
            weights = F.softmax(task_kls.detach() / self.adaptive_temp, dim=0) * num_old_tasks

        return (weights * task_kls).mean()

    def _compute_relational_distillation(self, new_features, old_features):
        """Pairwise-similarity-matrix distillation to preserve relative feature structure."""
        new_norm = F.normalize(new_features, p=2, dim=1)
        old_norm = F.normalize(old_features, p=2, dim=1)
        new_sim_matrix = torch.mm(new_norm, new_norm.t())
        old_sim_matrix = torch.mm(old_norm, old_norm.t())
        return F.mse_loss(new_sim_matrix, old_sim_matrix.detach())

    def _compute_feature_distillation(self, new_features, old_features):
        mse_loss = F.mse_loss(new_features, old_features)
        cosine_loss = 1 - F.cosine_similarity(new_features, old_features, dim=1).mean()
        relation_loss = self._compute_relational_distillation(new_features, old_features)
        return mse_loss + cosine_loss + self.lambda_relation * relation_loss

    def _apply_rotation(self, images, targets, t):
        images_90 = torch.rot90(images, 1, [2, 3])
        images_180 = torch.rot90(images, 2, [2, 3])
        images_270 = torch.rot90(images, 3, [2, 3])
        images_rot = torch.cat([images_90, images_180, images_270], dim=0)

        num_classes = self.model.task_cls[t]
        rot_labels_90 = targets + num_classes
        rot_labels_180 = targets + num_classes * 2
        rot_labels_270 = targets + num_classes * 3
        targets_rot = torch.cat([rot_labels_90, rot_labels_180, rot_labels_270], dim=0)

        images = torch.cat([images, images_rot], dim=0)
        targets = torch.cat([targets, targets_rot], dim=0)
        return images, targets

    def post_train_process(self, t, trn_loader):
        # The method invokes this hook twice per stage. Preserve only the first state,
        # which is the classifier immediately before prototype-based calibration.
        if self.capture_analysis and self.analysis_pre_calibration_task != t:
            self.analysis_pre_calibration_state = {
                key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
            }
            self.analysis_pre_calibration_task = t

        self._compute_prototypes(t, trn_loader)
        if t > 0:
            self._balance_head(trn_loader)


    def _compute_prototypes(self, t, trn_loader):
        self.model.eval()
        all_features, all_labels = [], []

        with torch.no_grad():
            for images, targets in trn_loader:
                images = images.to(self.device)
                features = self.model(images, return_features=True)[1]
                all_features.append(features.cpu())
                all_labels.append(targets.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(all_labels)


        for label in unique_labels:
            label = label.item()
            class_features = all_features[all_labels == label]
            n_class = class_features.shape[0]
            self.class_stats[label] = n_class

            use_multimodal = n_class >= self.min_samples_for_multimodal \
                and self.num_subprototypes > 1
            sub_clusters = self._kmeans_split(class_features, self.num_subprototypes) \
                if use_multimodal else [class_features]

            class_gaussians = []
            for sub_features in sub_clusters:
                n_sub = sub_features.shape[0]
                if n_sub == 0:
                    continue

                sub_mean = torch.mean(sub_features, dim=0)
                if n_sub > 1:
                    sub_cov = torch.cov(sub_features.T)
                else:
                    sub_cov = torch.zeros(self.feature_size, self.feature_size)

                sub_cov = sub_cov + self.covariance_epsilon * torch.eye(self.feature_size)

                gaussian = MultivariateNormal(sub_mean, covariance_matrix=sub_cov)
                class_gaussians.append({'gaussian': gaussian, 'weight': n_sub / n_class, 'n_samples': n_sub})

            self.gaussians[label] = class_gaussians
            self.prototypes.append(torch.mean(class_features, dim=0))
            self.prototype_labels.append(label)

    def _kmeans_split(self, features, k):
        """Lightweight K-means (cheap GMM approximation) to capture multi-modal class structure."""
        n = features.shape[0]
        k = min(k, n)
        if k <= 1:
            return [features]

        with torch.no_grad():
            perm = torch.randperm(n)[:k]
            centroids = features[perm].clone()

            for _ in range(10):
                dists = torch.cdist(features, centroids)
                assignments = torch.argmin(dists, dim=1)
                new_centroids = centroids.clone()
                for c in range(k):
                    cluster_mask = assignments == c
                    if cluster_mask.sum() > 0:
                        new_centroids[c] = features[cluster_mask].mean(dim=0)
                shift = (new_centroids - centroids).norm()
                centroids = new_centroids
                if shift < 1e-6:
                    break

            dists = torch.cdist(features, centroids)
            assignments = torch.argmin(dists, dim=1)

        return [features[assignments == c] for c in range(k) if (assignments == c).sum() > 0]

    def _balance_head(self, trn_loader):
        for param in self.model.model.parameters():
            param.requires_grad = False
        for head in self.model.heads:
            for param in head.parameters():
                param.requires_grad = True

        optimizer = torch.optim.SGD(
            [p for head in self.model.heads for p in head.parameters() if p.requires_grad],
            lr=self.balanced_lr, weight_decay=5e-4, momentum=0.9, foreach=False
        )
        criterion = nn.CrossEntropyLoss()

        current_features, current_targets = self._get_features(trn_loader)
        features_dataset = TensorDataset(current_features, current_targets)

        proto_features, proto_labels = self._generate_samples_from_prototypes()
        prototypes_dataset = TensorDataset(proto_features, proto_labels)

        complete_dataset = ConcatDataset([features_dataset, prototypes_dataset])

        for epoch in range(self.balanced_epochs):
            epoch_loader = DataLoader(complete_dataset, batch_size=self.balanced_bs, shuffle=True)
            epoch_loss = 0.0

            for features, targets in epoch_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = torch.cat([head(features) for head in self.model.heads], dim=1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f'    Epoch {epoch+1}/{self.balanced_epochs}, Loss: {epoch_loss/len(epoch_loader):.4f}')

        for param in self.model.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def _get_features(self, loader):
        self.model.eval()
        features_list, labels_list = [], []
        for images, targets in loader:
            images = images.to(self.device)
            features = self.model(images, return_features=True)[1]
            features_list.append(features.cpu())
            labels_list.append(targets.cpu())
        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

    @torch.no_grad()
    def _generate_samples_from_prototypes(self):
        """Sample synthetic old-class features from each class's sub-prototype Gaussians,
        proportionally to each sub-prototype's relative weight."""
        samples, labels = [], []

        for label, class_gaussians in self.gaussians.items():
            total_samples_for_class = self.class_stats[label]
            if len(class_gaussians) == 0:
                continue

            raw_counts = [w['weight'] * total_samples_for_class for w in class_gaussians]
            counts = [int(round(c)) for c in raw_counts]
            diff = total_samples_for_class - sum(counts)
            if diff != 0 and len(counts) > 0:
                counts[0] += diff
            counts = [max(c, 0) for c in counts]

            for sub_info, num_samples in zip(class_gaussians, counts):
                if num_samples <= 0:
                    continue
                class_samples = sub_info['gaussian'].sample((num_samples,))
                class_labels = torch.full((num_samples,), label, dtype=torch.long)
                samples.append(class_samples)
                labels.append(class_labels)

        if len(samples) == 0:
            return torch.empty(0, self.feature_size), torch.empty(0, dtype=torch.long)

        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()

            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                features = self.model(images, return_features=True)[1]

                offset = self.model.task_offset[t]
                total_loss_value = F.cross_entropy(outputs[t], targets - offset)

                if t > 0 and self.old_model is not None:
                    old_outputs = self.old_model(images)
                    old_features = self.old_model(images, return_features=True)[1]

                    if self.lambda_distill > 0:
                        total_loss_value += self.lambda_distill * self._compute_logit_distillation(
                            outputs, old_outputs, t, temperature=self.distill_temperature
                        )
                    if self.lambda_feat_distill > 0:
                        total_loss_value += self.lambda_feat_distill * self._compute_feature_distillation(
                            features, old_features
                        )

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets, t)
                total_loss += total_loss_value.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets, t):
        pred_taw = torch.argmax(outputs[t], dim=1) + self.model.task_offset[t]
        hits_taw = (pred_taw == targets).float()

        all_outputs = torch.cat([outputs[task] for task in range(t + 1)], dim=1)
        pred_tag = torch.argmax(all_outputs, dim=1)

        for task in range(t + 1):
            task_start = sum([self.model.task_cls[i] for i in range(task)])
            task_end = sum([self.model.task_cls[i] for i in range(task + 1)])
            task_mask = (pred_tag >= task_start) & (pred_tag < task_end)
            pred_tag[task_mask] += self.model.task_offset[task] - task_start

        hits_tag = (pred_tag == targets).float()
        return hits_taw, hits_tag
