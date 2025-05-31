from multiprocessing.sharedctypes import Value
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score 

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        audio_features,
        nodes_features,
        audio_features_mlp=None, 
        nodes_features_mlp=None,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False
):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_audio_features = hvd.allgather(audio_features)
            all_nodes_features = hvd.allgather(nodes_features)
            if mlp_loss:
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                all_nodes_features_mlp = hvd.allgather(nodes_features_mlp)
        else:
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)
                all_nodes_features = hvd.allgather(nodes_features)
                if mlp_loss:
                    all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                    all_nodes_features_mlp = hvd.allgather(nodes_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features = list(all_audio_features.chunk(world_size, dim=0))
                gathered_nodes_features = list(all_nodes_features.chunk(world_size, dim=0))
                gathered_audio_features[rank] = audio_features
                gathered_nodes_features[rank] = nodes_features
                all_audio_features = torch.cat(gathered_audio_features, dim=0)
                all_nodes_features = torch.cat(gathered_nodes_features, dim=0)
                if mlp_loss:
                    gathered_audio_features_mlp = list(all_audio_features_mlp.chunk(world_size, dim=0))
                    gathered_nodes_features_mlp = list(all_nodes_features_mlp.chunk(world_size, dim=0))
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_nodes_features_mlp[rank] = nodes_features_mlp
                    all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
                    all_nodes_features_mlp = torch.cat(gathered_nodes_features_mlp, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
            all_nodes_features = torch.cat(torch.distributed.nn.all_gather(nodes_features), dim=0)
            if mlp_loss:
                all_audio_features_mlp = torch.cat(torch.distributed.nn.all_gather(audio_features_mlp), dim=0)
                all_nodes_features_mlp = torch.cat(torch.distributed.nn.all_gather(nodes_features_mlp), dim=0)
        else:
            gathered_audio_features = [torch.zeros_like(audio_features) for _ in range(world_size)]
            gathered_nodes_features = [torch.zeros_like(nodes_features) for _ in range(world_size)]
            dist.all_gather(gathered_audio_features, audio_features)
            dist.all_gather(gathered_nodes_features, nodes_features)
            if mlp_loss:
                gathered_audio_features_mlp = [torch.zeros_like(audio_features_mlp) for _ in range(world_size)]
                gathered_nodes_features_mlp = [torch.zeros_like(nodes_features_mlp) for _ in range(world_size)]
                dist.all_gather(gathered_audio_features_mlp, audio_features_mlp)
                dist.all_gather(gathered_nodes_features_mlp, nodes_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features[rank] = audio_features
                gathered_nodes_features[rank] = nodes_features
                if mlp_loss:
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_nodes_features_mlp[rank] = nodes_features_mlp

            all_audio_features = torch.cat(gathered_audio_features, dim=0)
            all_nodes_features = torch.cat(gathered_nodes_features, dim=0)
            if mlp_loss:
                all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
                all_nodes_features_mlp = torch.cat(gathered_nodes_features_mlp, dim=0)
    if mlp_loss:
        return all_audio_features, all_nodes_features, all_audio_features_mlp, all_nodes_features_mlp
    else:
        return all_audio_features, all_nodes_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            mlp_loss=False,
            weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa!=0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, audio_features, nodes_features, logit_scale_a, logit_scale_t=None, audio_features_mlp=None, nodes_features_mlp=None):
        device = audio_features.device
        if self.mlp_loss:
            if self.world_size > 1:
                all_audio_features, all_nodes_features, all_audio_features_mlp, all_nodes_features_mlp = gather_features(
                    audio_features=audio_features,nodes_features=nodes_features,
                    audio_features_mlp=audio_features_mlp,nodes_features_mlp=nodes_features_mlp,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )
                if self.local_loss:
                    a_logits_per_audio = logit_scale_a * audio_features @ all_nodes_features_mlp.T
                    a_logits_per_nodes = logit_scale_a * nodes_features_mlp @ all_audio_features.T
                    t_logits_per_audio = logit_scale_t * audio_features_mlp @ all_nodes_features.T
                    t_logits_per_nodes = logit_scale_t * nodes_features @ all_audio_features_mlp.T
                else:
                    a_logits_per_audio = logit_scale_a * all_audio_features @ all_nodes_features_mlp.T
                    a_logits_per_nodes = a_logits_per_audio.T
                    t_logits_per_audio = logit_scale_t * all_audio_features_mlp @ all_nodes_features.T
                    t_logits_per_nodes = t_logits_per_audio.T
            else:
                a_logits_per_audio = logit_scale_a * audio_features @ nodes_features_mlp.T
                a_logits_per_nodes = logit_scale_a * nodes_features_mlp @ audio_features.T
                t_logits_per_audio = logit_scale_t * audio_features_mlp @ nodes_features.T
                t_logits_per_nodes = logit_scale_t * nodes_features @ audio_features_mlp.T

            # calculated ground-truth and cache if enabled
            num_logits = a_logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels) +
                    F.cross_entropy(a_logits_per_nodes, labels) + 
                    F.cross_entropy(t_logits_per_audio, labels) +
                    F.cross_entropy(t_logits_per_nodes, labels) 
                    ) / 4
            else:
                audio_weight = (audio_features@audio_features.T).detach()
                audio_weight = (torch.exp(torch.sum(audio_weight, axis=1)/(self.weight_loss_kappa*len(audio_weight)))).detach()
                nodes_weight = (nodes_features@nodes_features.T).detach()
                nodes_weight = (torch.exp(torch.sum(nodes_weight, axis=1)/(self.weight_loss_kappa*len(nodes_features)))).detach()
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels, weight=audio_weight) +
                    F.cross_entropy(a_logits_per_nodes, labels, weight=audio_weight) + 
                    F.cross_entropy(t_logits_per_audio, labels, weight=nodes_weight) +
                    F.cross_entropy(t_logits_per_nodes, labels, weight=nodes_weight) 
                    ) / 4
        else:
            if self.world_size > 1:
                all_audio_features, all_nodes_features = gather_features(
                    audio_features=audio_features,nodes_features=nodes_features,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )

                if self.local_loss:
                    logits_per_audio = logit_scale_a * audio_features @ all_nodes_features.T
                    logits_per_nodes = logit_scale_a * nodes_features @ all_audio_features.T
                else:
                    logits_per_audio = logit_scale_a * all_audio_features @ all_nodes_features.T
                    logits_per_nodes = logits_per_audio.T
            else:
                #print("calcolando loss")
                logits_per_audio = logit_scale_a * audio_features @ nodes_features.T
                logits_per_nodes = logit_scale_a * nodes_features @ audio_features.T
                #print("logits_per_audio: ",logits_per_audio)
                #print("logits_per_nodes: ",logits_per_nodes)
                #print("logits_per_audio.shape: ",logits_per_audio.shape)
                #print("logits_per_nodes.shape: ",logits_per_nodes.shape)
                #print("logits_per_audio.mean: ",torch.mean(logits_per_audio))
                #print("logits_per_nodes.mean: ",torch.mean(logits_per_nodes))

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels) +
                    F.cross_entropy(logits_per_nodes, labels)
                    ) / 2
            else:
                audio_weight = (all_audio_features@all_audio_features.T).detach()
                audio_weight = (torch.exp(torch.sum(audio_weight, axis=1)/(self.weight_loss_kappa*len(all_audio_features)))).detach()
                nodes_weight = (all_nodes_features@all_nodes_features.T).detach()
                nodes_weight = (torch.exp(torch.sum(nodes_weight, axis=1)/(self.weight_loss_kappa*len(all_nodes_features)))).detach()
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels, weight=nodes_weight) +
                    F.cross_entropy(logits_per_nodes, labels, weight=audio_weight)
                    ) / 2
        return total_loss

def lp_gather_features(
        pred,
        target,
        world_size=1,
        use_horovod=False
):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        with torch.no_grad():
            all_preds = hvd.allgather(pred)
            all_targets = hvd.allgath(target)
    else:
        gathered_preds = [torch.zeros_like(pred) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(target) for _ in range(world_size)]

        dist.all_gather(gathered_preds, pred)
        dist.all_gather(gathered_targets, target)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)

    return all_preds, all_targets


def get_map(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))

def get_acc(pred, target):
    pred = torch.argmax(pred,1).numpy()
    target = torch.argmax(target,1).numpy()
    return accuracy_score(target, pred)

def get_mauc(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(roc_auc_score(target, pred, average=None))


class LPMetrics(object):
    def __init__(self, metric_names = ['map','acc','mauc']):
        self.metrics = []
        for name in metric_names:
            self.metrics.append(self.get_metric(name))
        self.metric_names = metric_names

    def get_metric(self,name):
        if name == 'map':
            return get_map
        elif name == 'acc':
            return get_acc
        elif name == 'mauc':
            return get_mauc
        else:
            raise ValueError(f'the metric should be at least one of [map, acc, mauc]')

    def evaluate_mertics(self, pred, target):
        metric_dict = {}
        for i in range(len(self.metric_names)):
            metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
        return metric_dict


def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)


class LPLoss(nn.Module):

    def __init__(self, loss_name):
        super().__init__()
        if loss_name == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == 'ce':
            self.loss_func = calc_celoss
        elif loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f'the loss func should be at least one of [bce, ce, mse]')

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss
        