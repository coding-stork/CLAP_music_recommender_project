import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

try:
    import wandb
except ImportError:
    wandb = None

from clap_module import ClipLoss, gather_features
from .distributed import is_master


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model
        
        
total_nodes = []

old_uri_to_new_id = {}
with open("/content/old_uri_to_new_id.pkl","rb") as f:
  old_uri_to_new_id = pickle.load(f)

old_id_to_new_id = {}
with open("/content/old_id_to_new_id.pkl","rb") as f:
  old_id_to_new_id = pickle.load(f)

with open('/content/biggest_total_nodes.pkl', 'rb') as f:
    total_nodes = pickle.load(f)

import pandas as pd

orig_tracks_dataset = pd.read_csv("laion_clap/Relphormer/nva_final_tracks_less.csv")
new_tracks_dataset = pd.read_csv("laion_clap/Relphormer/more_tracks_less_modified.csv")
tracks_dataset = orig_tracks_dataset.merge(new_tracks_dataset,how="outer")
playlists_dataset = pd.read_csv("/content/final_playlists.csv")
artists_dataset = pd.read_csv("/content/artists.csv")
nomi_tracce = tracks_dataset['name'].values
data_tracce = tracks_dataset['release_date'].values
uri_tracce = tracks_dataset['track_uri'].values
track_indexes_for_playlists = {}
track_indexes_for_artists = {}
total_artists = []
total_playlists = []
total_titles = []

for i, row in tracks_dataset.iterrows():
  nome_traccia = row['name']
  nome_artisti = row['artists_names']
  playlist_uris = row['playlist_uris']
  nome_artisti = nome_artisti.strip('][').strip("'").strip('"').split("', '")

  for k in nome_artisti:
    if k not in track_indexes_for_artists:
      track_indexes_for_artists[k] = []
      total_artists.append(k)
    track_indexes_for_artists[k].append(i)

  nome_artisti = ", ".join(nome_artisti)
  print(nome_artisti)
  playlist_uris = playlist_uris.strip('][').strip("'").split("', '")
  total_titles.append(nome_artisti+" - "+nome_traccia)

def find_top_10_audio(node_tensor, all_audio_embeddings):

    audio_keys = list(all_audio_embeddings.keys())
    audio_embeddings = np.stack([all_audio_embeddings[key] for key in audio_keys])
    similarities = cosine_similarity(audio_embeddings, [node_tensor])
    top_10_indices = np.argsort(np.squeeze(similarities, axis=1))[-20:][::-1]
    top_10_audio_keys = [audio_keys[idx] for idx in top_10_indices]
    return top_10_audio_keys

def find_top_10_nodes(audio_tensor, all_node_embeddings):

    node_keys = list(all_node_embeddings.keys())
    node_embeddings = np.stack([all_node_embeddings[key] for key in node_keys])
    similarities = cosine_similarity(node_embeddings, [audio_tensor])
    top_10_indices = np.argsort(np.squeeze(similarities, axis=1))[-20:][::-1]
    top_10_node_keys = [node_keys[idx] for idx in top_10_indices]
    return top_10_node_keys
    
old_track_uris = orig_tracks_dataset['track_uri']
new_track_uris = new_tracks_dataset['track_uri']

orig_order_track_uris = list(np.concatenate((old_track_uris,new_track_uris)))

def train_one_epoch(
        model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None
):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        mlp_loss=args.clap_mlploss,
        weight_loss_kappa=args.kappa,
    )

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # for toy dataset
    if args.dataset_type == "toy":
        dataloader.dataset.generate_queue()

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        # logging.info(f"batch {i} of {num_batches_per_epoch}")

        #print("i: ",i)

        step = num_batches_per_epoch * epoch + i
        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(step)
        else:
            scheduler(step)

        audios = batch  # contains mel_spec, wavform, and longer list
        relphormer_inputs = batch['relphormer_inputs']
                
        #print("batch in train: ",batch)
        #more operations for relphormer inputs that are taken from pytorch_lightning model:

        labels = relphormer_inputs.pop("labels")
        label = relphormer_inputs.pop("label")
        input_ids = relphormer_inputs['input_ids']
        
        distance_attention = relphormer_inputs.pop("distance_attention")
        graph_attn_bias = distance_attention
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, model.graph_branch.num_heads, 1, 1)
        graph_attn_bias[:, :, 0, :] = 0
        graph_attn_bias[:, :, :, 0] = 0
        relphormer_inputs["distance_attention"] = graph_attn_bias
        relphormer_inputs["return_dict"] = True

        relphormer_inputs = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in relphormer_inputs.items()
        }        

        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()

        with autocast():
            (
                audio_features,
                nodes_features,
                audio_features_mlp,
                nodes_features_mlp,
                logit_scale_a,
                logit_scale_t,
            ) = model(audios, relphormer_inputs, device)

            if args.clap_mlploss:
                total_loss = loss(
                    audio_features=audio_features,
                    nodes_features=nodes_features,
                    logit_scale_a=logit_scale_a,
                    logit_scale_t=logit_scale_t,
                    audio_features_mlp=audio_features_mlp,
                    nodes_features_mlp=nodes_features_mlp
                )
            else:
                total_loss = loss(
                    audio_features=audio_features,
                    nodes_features=nodes_features,
                    logit_scale_a=logit_scale_a
                )


        if isinstance(optimizer, dict):
            if scaler is not None:
                scaler.scale(total_loss).backward()
                #torch.nn.utils.clip_grad_norm_(model.graph_branch.parameters(), 5)
                for o_ in optimizer.values():
                    if args.horovod:
                        o_.synchronize()
                        scaler.unscale_(o_)
                        with o_.skip_synchronize():
                            scaler.step(o_)
                    else:
                        scaler.step(o_)
                scaler.update()
            else:
                total_loss.backward()
                for o_ in optimizer.values():
                    o_.step()
        else:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                #torch.nn.utils.clip_grad_norm_(model.graph_branch.parameters(), 5)
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.graph_branch.parameters(), 5)
                optimizer.step()

        #for name, param in model.graph_branch.named_parameters():
        #  if param.grad is not None:
        #      print(f"Gradient for {name}: {param.grad.abs().mean()}")
        #  else:
        #      print(f"No gradient for {name}")

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale_a.clamp_(0, math.log(100))
            if args.clap_mlploss:
                unwrap_model(model).logit_scale_t.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            if isinstance(audios, dict):
                batch_size = len(audios["waveform"])
            else:
                batch_size = len(audios)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar_a = logit_scale_a.item()
            logit_scale_scalar_t = logit_scale_t.item()
            if isinstance(optimizer, dict):
                if args.clap_mlploss:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f} "
                        f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                        f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                        f"Logit Scale Text: {logit_scale_scalar_t:.3f}"
                    )
                    log_data = {
                        "loss": loss_m.val,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "scale_audio": logit_scale_scalar_a,
                        "scale_text": logit_scale_scalar_t,
                        "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                    }
                else:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f} "
                        f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                        f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                    )
                    log_data = {
                        "loss": loss_m.val,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "scale_audio": logit_scale_scalar_a,
                        "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                    }

            else:
                if args.clap_mlploss:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f} "
                        f"LR: {optimizer.param_groups[0]['lr']:5f} "
                        f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                        f"Logit Scale Text: {logit_scale_scalar_t:.3f}"
                    )

                    # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                    log_data = {
                        "loss": loss_m.val,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "scale_audio": logit_scale_scalar_a,
                        "scale_text": logit_scale_scalar_t,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                else:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f} "
                        f"LR: {optimizer.param_groups[0]['lr']:5f} "
                        f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                    )

                    # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                    log_data = {
                        "loss": loss_m.val,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "scale_audio": logit_scale_scalar_a,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                '''if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step})'''

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()

    all_audio_features = {}
    all_nodes_features = {}    

    # CHANGE
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # metrics.update(zero_shot_metrics)
    if is_master(args):
        print('Evaluating...')
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    if args.val_dataset_names == ['Clotho', 'audiocaps']:
        # if only clotho and audiocaps are used, then we will use a different evaluation function.
        # This is because in the Clotho and audiocaps valid and test set, there are 5 text for 1 audio.
        if args.parallel_eval:
            # (yusong): just a hack here. Don't use parallel eval when evaluating only clotho and audiocaps.
            raise NotImplementedError("Parallel evaluation not supported for eval only Clotho and audiocaps.")
        val_metrics_per_dataset = evaluate_clotho_audiocaps(model, data, epoch, args, autocast, device, tb_writer)
        for m in val_metrics_per_dataset.values():
            metrics.update(m)
        if "epoch" not in metrics.keys():
            metrics.update({"epoch": epoch})
        metrics = select_top_metric_clotho_audiocaps(metrics, val_metrics_per_dataset, args)
    elif "val" in data and (
            args.val_frequency
            and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_audio_features @ all_text_features will blow up memory and compute very quickly
        eval_info = {}
        if args.clap_mlploss:
            eval_info["all"] = {
                "cumulative_loss": 0.0,
                "num_samples": 0,
                "all_audio_features": [],
                "all_text_features": [],
                "all_audio_features_mlp": [],
                "all_text_features_mlp": []
            }  # cumulative_loss = 0.0
        else:
            eval_info["all"] = {
                "cumulative_loss": 0.0,
                "num_samples": 0,
                "all_audio_features": [],
                "all_text_features": []
            }  # cumu
        # all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp = [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audios = batch  # contains mel_spec, wavform, and longer list
                relphormer_inputs = batch['relphormer_inputs']
                
                #print("batch in train: ",batch)

                labels = relphormer_inputs.pop("labels")
                input_ids = relphormer_inputs['input_ids']
                # single label
                label = relphormer_inputs.pop('label')
        
        
                my_keys = list(relphormer_inputs.keys())
                for k in my_keys:
                    if k not in ["input_ids", "attention_mask", "token_type_ids", "distance_attention"]:
                        relphormer_inputs.pop(k)
                        
                distance_attention = relphormer_inputs.pop("distance_attention")
                graph_attn_bias = distance_attention
                #print("model.graph_branch.num_heads: ",model.graph_branch.num_heads)
                graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, model.graph_branch.num_heads, 1, 1)
                graph_attn_bias[:, :, 0, :] = 0
                graph_attn_bias[:, :, :, 0] = 0
                
                relphormer_inputs["distance_attention"] = graph_attn_bias
                relphormer_inputs["return_dict"] = True


                relphormer_inputs = {
                    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                    for key, value in relphormer_inputs.items()
                }

                
                all_names = list(set(["-".join(b.split("/")[-3:-1]) for b in batch['__url__']]))
                for name in all_names:
                    if name not in eval_info.keys():
                        if args.clap_mlploss:
                            eval_info[name] = {
                                "cumulative_loss": 0.0,
                                "num_samples": 0,
                                "all_audio_features": [],
                                "all_text_features": [],
                                "all_audio_features_mlp": [],
                                "all_text_features_mlp": [],
                            }
                        else:
                            eval_info[name] = {
                                "cumulative_loss": 0.0,
                                "num_samples": 0,
                                "all_audio_features": [],
                                "all_text_features": []
                            }
                with autocast():
                    (
                        audio_features,
                        nodes_features,
                        audio_features_mlp,
                        nodes_features_mlp,
                        logit_scale_a,
                        logit_scale_t,
                    ) = model(audios, relphormer_inputs, device)
                    
                    list_audio_features = torch.clone(audio_features).cpu().numpy()
                    list_nodes_features = torch.clone(nodes_features).cpu().numpy()

                    for k, a in zip(batch['__key__'],list_audio_features):
                        all_audio_features[k] = a

                    for k, n in zip(batch['__key__'],list_nodes_features):
                        all_nodes_features[k] = n

                    if args.parallel_eval:
                        # multi-GPU eval
                        if args.clap_mlploss:
                            (
                                audio_features,
                                nodes_features,
                                audio_features_mlp,
                                nodes_features_mlp,
                            ) = gather_features(
                                audio_features=audio_features,
                                nodes_features=nodes_features,
                                audio_features_mlp=audio_features_mlp,
                                nodes_features_mlp=nodes_features_mlp,
                                local_loss=False,
                                gather_with_grad=False,
                                rank=args.rank,
                                world_size=args.world_size,
                                use_horovod=args.horovod,
                                mlp_loss=args.clap_mlploss
                            )
                        else:
                            (
                                audio_features,
                                nodes_features,
                            ) = gather_features(
                                audio_features=audio_features,
                                nodes_features=nodes_features,
                                local_loss=False,
                                gather_with_grad=False,
                                rank=args.rank,
                                world_size=args.world_size,
                                use_horovod=args.horovod,
                                mlp_loss=args.clap_mlploss
                            )

                    if is_master(args):
                        num_samples += audio_features.shape[0]
                        for n in [*all_names, "all"]:
                            if n == "all":
                                eval_info[n]["all_audio_features"].append(
                                    audio_features.cpu()
                                )
                                eval_info[n]["all_text_features"].append(
                                    nodes_features.cpu()
                                )
                                if args.clap_mlploss:
                                    eval_info[n]["all_audio_features_mlp"].append(
                                        audio_features_mlp.cpu()
                                    )
                                    eval_info[n]["all_text_features_mlp"].append(
                                        nodes_features_mlp.cpu()
                                    )
                            else:
                                idx = np.where(
                                    np.array(
                                        ["-".join(b.split("/")[-3:-1]) for b in batch['__url__']]
                                    )
                                    == n
                                )[0]
                                eval_info[n]["all_audio_features"].append(
                                    audio_features.cpu().index_select(
                                        0, torch.tensor(idx).long()
                                    )
                                )
                                eval_info[n]["all_text_features"].append(
                                    nodes_features.cpu().index_select(
                                        0, torch.tensor(idx).long()
                                    )
                                )
                                if args.clap_mlploss:
                                    eval_info[n]["all_audio_features_mlp"].append(
                                        audio_features_mlp.cpu().index_select(
                                            0, torch.tensor(idx).long()
                                        )
                                    )
                                    eval_info[n]["all_text_features_mlp"].append(
                                        nodes_features_mlp.cpu().index_select(
                                            0, torch.tensor(idx).long()
                                        )
                                    )
                        #  print(f'eval step {i}') #  (yusong): for debug

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                if is_master(args) and (i % 100) == 0:  # and i != 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
                    )
                    
            #do some evaluations to check if the model is working
            
            print("EVALUATION MEGADETH. NODES\n\n")
            
            #WITH NODES

            new_num = "8121_1.flac"
            int_new_num = int(new_num[:-7])
            print(int_new_num)

            real_new_num = old_id_to_new_id[int_new_num]
            real_new_num = str(real_new_num) + new_num[-7:-5] 
            
            print(all_nodes_features.keys())
            results = find_top_10_audio(all_nodes_features[real_new_num],all_audio_features)
            print(results)
            
            print(total_titles[old_id_to_new_id[int_new_num]])
            
            for i in results:
                print(total_titles[old_uri_to_new_id[orig_order_track_uris[int(i[:-2])]]])
              
            #WITH AUDIO
            
            print("EVALUATION MEGADETH. AUDIOS\n\n")

            new_num = "8121_1.flac"
            int_new_num = int(new_num[:-7])
            print(int_new_num)

            real_new_num = old_id_to_new_id[int_new_num]
            real_new_num = str(real_new_num) + new_num[-7:-5] 
            
            results = find_top_10_nodes(all_audio_features[real_new_num],all_nodes_features)
            print(results)
            
            print(total_titles[old_id_to_new_id[int_new_num]])
            
            for i in results:
                print(total_titles[old_uri_to_new_id[orig_order_track_uris[int(i[:-2])]]])
              
            print("EVALUATION SPACE LACES. NODES\n\n")
             
            #WITH NODES

            new_num = "10867_2.flac"
            int_new_num = int(new_num[:-7])
            print(int_new_num)

            results = find_top_10_audio(all_nodes_features[new_num[:-5]],all_audio_features)
            print(results)
            
            print(total_titles[int_new_num])
            
            for i in results:
                print(total_titles[old_uri_to_new_id[orig_order_track_uris[int(i[:-2])]]])
                
            #not always necessary but it's better to not risk
            if os.path.isfile("/content/relphormer_biggest_all_node_embeddings.pkl"):
                os.remove("/content/relphormer_biggest_all_node_embeddings.pkl")
            if os.path.isfile("/content/relphormer_biggest_all_audio_embeddings.pkl"):
                os.remove("/content/relphormer_biggest_all_audio_embeddings.pkl")

            #save the embeddings in pickles
            with open("/content/relphormer_biggest_all_node_embeddings.pkl","wb") as f:
                pickle.dump(all_nodes_features,f)

            with open("/content/relphormer_biggest_all_audio_embeddings.pkl","wb") as f:
                pickle.dump(all_audio_features,f)

            del all_nodes_features
            del all_audio_features                    
                    
            if is_master(args):
                val_metrics_per_dataset = {}
                _eval_chunk_size = 2000  # Adjust based on memory limits

                for n in eval_info.keys():
                    _audio_features = torch.cat(eval_info[n]["all_audio_features"])
                    _text_features = torch.cat(eval_info[n]["all_text_features"])
                    
                    if args.clap_mlploss:
                        _audio_features_mlp = torch.cat(eval_info[n]["all_audio_features_mlp"])
                        _text_features_mlp = torch.cat(eval_info[n]["all_text_features_mlp"])
                    
                    num_samples = _audio_features.shape[0]
                    _metrics_accumulator = []

                    for _chunk_start in range(0, num_samples, _eval_chunk_size):
                        _chunk_end = min(_chunk_start + _eval_chunk_size, num_samples)

                        _audio_chunk = _audio_features[_chunk_start:_chunk_end]
                        _text_chunk = _text_features[_chunk_start:_chunk_end]

                        if args.clap_mlploss:
                            _audio_mlp_chunk = _audio_features_mlp[_chunk_start:_chunk_end]
                            _text_mlp_chunk = _text_features_mlp[_chunk_start:_chunk_end]

                            _chunk_metrics = get_metrics(
                                audio_features=_audio_chunk,
                                text_features=_text_chunk,
                                logit_scale_a=logit_scale_a.cpu(),
                                audio_features_mlp=_audio_mlp_chunk,
                                text_features_mlp=_text_mlp_chunk,
                                logit_scale_t=logit_scale_t.cpu(),
                                mlp_loss=args.clap_mlploss
                            )
                        else:
                            _chunk_metrics = get_metrics(
                                audio_features=_audio_chunk,
                                text_features=_text_chunk,
                                logit_scale_a=logit_scale_a.cpu(),
                                mlp_loss=args.clap_mlploss
                            )

                        _metrics_accumulator.append(_chunk_metrics)

                    # Aggregate metrics across all chunks
                    metrics_single_dataset = {}
                    for key in _metrics_accumulator[0].keys():
                        metrics_single_dataset[key] = sum(m[key] for m in _metrics_accumulator) / len(_metrics_accumulator)

                    val_metrics_per_dataset[n] = {n + "/" + k: v for k, v in metrics_single_dataset.items()}
                    metrics.update(val_metrics_per_dataset[n])

                if "epoch" not in metrics.keys():
                    metrics.update({"epoch": epoch})

            if is_master(args):
                if not metrics:
                    return metrics

                logging.info(
                    f"Eval Epoch: {epoch} "
                    + "\n".join(
                        [
                            "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in m.items()])
                            for m in val_metrics_per_dataset.values()
                        ]
                    )
                )


        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        '''if args.wandb:
            assert wandb is not None, "Please install wandb."
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, "epoch": epoch})'''

        return metrics
    else:
        return metrics


def get_metrics(
        audio_features,
        text_features,
        logit_scale_a,
        audio_features_mlp=None,
        text_features_mlp=None,
        logit_scale_t=None,
        mlp_loss=False
):
    metrics = {}
    if mlp_loss:
        # Set up audio to text & text to audio similary matrice
        a_logits_per_audio = (
            (logit_scale_a * audio_features @ text_features_mlp.t()).detach().cpu()
        )
        a_logits_per_text = a_logits_per_audio.t().detach().cpu()
        t_logits_per_audio = (
            (logit_scale_t * audio_features_mlp @ text_features.t()).detach().cpu()
        )
        t_logits_per_text = t_logits_per_audio.t().detach().cpu()

        labels = torch.arange(audio_features.shape[0]).long()
        # Change the loss from two terms into four terms with 2x2 combined CE loss
        total_loss = (
                             F.cross_entropy(a_logits_per_audio, labels)
                             + F.cross_entropy(a_logits_per_text, labels)
                             + F.cross_entropy(t_logits_per_audio, labels)
                             + F.cross_entropy(t_logits_per_text, labels)
                     ) / 4

        metrics[f"cumulative_loss"] = total_loss.item()
        metrics[f"num_samples"] = audio_features.shape[0]

        logits = {
            "audio_to_text": (a_logits_per_audio + t_logits_per_audio) / 2,
            "text_to_audio": (a_logits_per_text + t_logits_per_text) / 2,
        }
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

    else:
        # print("text_features", text_features)
        # print("text_features.shape", text_features.shape)
        logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_audio.t().detach().cpu()

        labels = torch.arange(audio_features.shape[0]).long()
        # Change the loss from two terms into four terms with 2x2 combined CE loss
        total_loss = (
                             F.cross_entropy(logits_per_audio, labels)
                             + F.cross_entropy(logits_per_text, labels)
                     ) / 2

        metrics[f"cumulative_loss"] = total_loss.item()
        metrics[f"num_samples"] = audio_features.shape[0]

        logits = {"audio_to_text": logits_per_audio, "text_to_audio": logits_per_text}

        ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"{name}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    return metrics


def evaluate_clotho_audiocaps(
        model, data, epoch, args, autocast, device, tb_writer=None
):
    """
    Adapted from https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/utils.py.
    1. for text-to-audio retrieval, do 5 times and average the results
    2. for R@1, R@5, R@10 in audio-to-text retrieval, take the best rank among 5 text
    3. for map@10 in audio-to-text retrieval:
        3.1: sort the rank of 5 text
        3.2: exclude the rank >=10 (0-index)
        3.3: compute the map regarding the remaining ranks: np.mean(np.arange(1, len(ranks)+1) / ranks).
        (3.3) That is, take the top ranks of 5 text that is < 10, and assign the descending number as ground truth.
        (3.3) E.g.: the ground truth of first rank of the 5 text should be 1, the second rank should be 2, etc.
    """
    # TODO: (yusong) only support single GPU evaluation and only support non-mlp case for now.
    dataloader = data["val"].dataloader
    with torch.no_grad():
        eval_info = {}
        for i, batch in enumerate(dataloader):
            audios = batch  # contains mel_spec, wavform, and longer list

            # each item in the list has 5 texts
            if args.tmodel == "transformer":
                from clap_module import tokenize
                texts = [tokenize(t) for t in batch['full_text']]
                texts = torch.cat(texts)
            else:
                from .data import tokenizer
                texts = [tokenizer(t, tmodel=args.tmodel) for t in batch['full_text']]  # 5 texts for each audio
                texts = {k: torch.cat([t[k] for t in texts]) for k in texts[0].keys()}  # 5 x batch

            # audios = audios.to(device=device, non_blocking=True)

            # batch['__url__'] contains the path to the data tar this sample is from
            # So, b.split("/")[-3:-1] will get you '<dataset_name>-<dataset-split>'
            all_names = list(set(["-".join(b.split("/")[-3:-1]) for b in batch['__url__']]))
            for name in all_names:
                if name not in eval_info.keys():
                    # we will not use mlp outputs even if args.clap_mlploss=True
                    eval_info[name] = {
                        "cumulative_loss": 0.0,
                        "num_samples": 0,
                        "all_audio_features": [],
                        "all_text_features": []
                    }
            with autocast():
                audio_features = model(audios, None, None, None, device)
                text_features = model(None, input_ids, attention_mask, relformer_f_args, device)
                audio_features = F.normalize(audio_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                all_names = list(set(["-".join(b.split("/")[-3:-1]) for b in batch['__url__']]))
                for n in all_names:
                    idx = np.where(
                        np.array(
                            ["-".join(b.split("/")[-3:-1]) for b in batch['__url__']]
                        )
                        == n
                    )[0]
                    eval_info[n]["all_audio_features"].append(
                        audio_features.cpu().index_select(
                            0, torch.tensor(idx).long()
                        )
                    )
                    # (yusong) please double-check. This is for selecting 5 text features at once.
                    # because idx is a list of indices in size of num_samples,
                    # and text_features is a tensor of size (5*num_samples, dim)
                    # so we need to select 5 consecutive indices at once for a single index in idx.
                    eval_info[n]["all_text_features"].append(
                        text_features.cpu().reshape([-1, 5, text_features.shape[1]]).index_select(
                            0, torch.tensor(idx).long()
                        ).reshape([-1, text_features.shape[1]])
                    )

        val_metrics_all = {}

        for n in eval_info.keys():
            logit_scale_a, logit_scale_t = model(None, None, device)
            logit_scale_a = logit_scale_a.cpu()

            audio_features = torch.cat(eval_info[n]["all_audio_features"], dim=0)
            text_features = torch.cat(eval_info[n]["all_text_features"], dim=0)

            logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
            logits_per_text = logits_per_audio.t().detach().cpu()

            # logits_per_audio shape: [num_samples, num_samples*5]
            # logits_per_text shape: [num_samples*5, num_samples]

            logging.info(f"dataset {n}, logits_per_audio shape: {logits_per_audio.shape}, "
                         f"logits_per_text shape: {logits_per_text.shape}")

            metrics = {}
            num_samples = audio_features.shape[0]
            metrics[f"num_samples"] = num_samples

            # (yusong) the following code is very important, please double-check:
            # logits_per_audio.reshape(num_samples, num_samples, 5)[:, :, d]
            # logits_per_text.reshape(num_samples, 5, num_samples)[:, d, :]
            # Those two are retrieving one of the 5 text for each audio.
            labels = torch.arange(audio_features.shape[0]).long()
            audio_to_text_loss = [
                F.cross_entropy(
                    logits_per_audio.reshape(num_samples, num_samples, 5)[:, :, d], labels) for d in range(5)
            ]
            text_to_audio_loss = [
                F.cross_entropy(
                    logits_per_text.reshape(num_samples, 5, num_samples)[:, d, :], labels) for d in range(5)
            ]
            total_loss = (
                                 np.mean(audio_to_text_loss) + np.mean(text_to_audio_loss)
                         ) / 2

            metrics[f"cumulative_loss"] = total_loss.item()

            # text to audio: do 5 times
            pred_text = []
            for d in range(5):
                logit = logits_per_text.reshape(num_samples, 5, num_samples)[:, d, :]
                ground_truth = torch.arange(len(logit)).view(-1, 1)
                ranking = torch.argsort(logit, descending=True)  # [num_samples, num_samples]
                preds = torch.where(ranking == ground_truth)[1]
                pred_text.append(preds.detach().cpu().numpy())
            pred_text_concat = np.concatenate(pred_text, axis=0)  # [5*num_samples]
            metrics[f"text_to_audio_mean_rank"] = pred_text_concat.mean() + 1
            metrics[f"text_to_audio_median_rank"] = np.floor(np.median(pred_text_concat)) + 1
            for k in [1, 5, 10]:
                metrics[f"text_to_audio_R@{k}"] = np.mean(pred_text_concat < k)
            # map@10
            metrics[f"text_to_audio_mAP@10"] = np.mean(np.where(pred_text_concat < 10, 1 / (pred_text_concat + 1), 0.0))

            # audio to text: take the best result
            # for audio to text map 10, sort and assign descending ground truth.
            # see https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/utils.py#L103
            # map@10
            map_all = []
            pred_audio_all = []
            for d in range(num_samples):
                # logits_per_audio: [num_samples, num_samples*5]
                logit_single = logits_per_audio[d, :]  # [5*num_samples]
                # Ground-truth index: [d*5, d*5+1, d*5+2, d*5+3, d*5+4]
                ranking = torch.argsort(logit_single, descending=True)  # [5*num_samples]
                # ranking: the index of first match, second match, ...
                ground_truth = torch.arange(d * 5, d * 5 + 5)[None]
                all_pred = torch.where(torch.stack([ranking] * 5) == ground_truth.view(-1, 1))[1]
                min_pred = torch.min(all_pred)
                pred_audio_all.append(min_pred.detach().cpu().numpy())
                all_pred_filter = all_pred[all_pred < 10].detach().cpu().numpy()
                # /5 because we have 5 text, so it means for the text rank >=10 we count as 0.
                map_single = np.sum((np.arange(1, len(all_pred_filter) + 1) / (all_pred_filter + 1))) / 5
                map_all.append(map_single)
            metrics[f"audio_to_text_mAP@10"] = np.mean(map_all)
            for k in [1, 5, 10]:
                metrics[f"audio_to_text_R@{k}"] = np.mean(np.array(pred_audio_all) < k)

            val_metrics_all[n] = {
                n + "/" + k: v for k, v in metrics.items()
            }
    return val_metrics_all


def calculate_selection_performance_clotho_audiocaps(val_metrics_per_dataset):
    """
    Calculate performance for Clotho+AudioCaps for model selection.
    """
    selection_performance_all = []
    for n in val_metrics_per_dataset.keys():
        selection_performance = (val_metrics_per_dataset[n][f"{n}/audio_to_text_mAP@10"] +
                                 val_metrics_per_dataset[n][f"{n}/text_to_audio_mAP@10"]) / 2
        selection_performance_all.append(selection_performance)
    return np.mean(selection_performance_all)


def select_top_metric_clotho_audiocaps(metrics, val_metrics_per_dataset, args):
    # val_metrics_per_dataset: dict, key: dataset name, value: dict, key: metric name, value: metric value
    # metrics: dict, key: metric name, value: metric value
    # Hack: use args to save the top performance
    if not hasattr(args, "top_selection_performance"):
        selection_performance = calculate_selection_performance_clotho_audiocaps(val_metrics_per_dataset)
        # TODO: write the if and else together
        metric_update = {}
        for n in val_metrics_per_dataset.keys():
            for k in val_metrics_per_dataset[n].keys():
                metric_update[k.split('/')[0] + '-top' + '/' + k.split('/')[1]] = val_metrics_per_dataset[n][k]
        metric_update['top_selection_performance'] = selection_performance
        metric_update['top-selection-epoch'] = metrics['epoch']
        metrics.update(metric_update)
        args.top_metric = metric_update
        args.top_selection_performance = selection_performance
    else:
        selection_performance_new = calculate_selection_performance_clotho_audiocaps(val_metrics_per_dataset)
        selection_performance_old = args.top_selection_performance
        if selection_performance_new > selection_performance_old:
            metric_update = {}
            for n in val_metrics_per_dataset.keys():
                for k in val_metrics_per_dataset[n].keys():
                    metric_update[k.split('/')[0] + '-top' + '/' + k.split('/')[1]] = val_metrics_per_dataset[n][k]
            metric_update['top_selection_performance'] = selection_performance_new
            metric_update['top-selection-epoch'] = metrics['epoch']
            metrics.update(metric_update)
            args.top_metric = metric_update
            args.top_selection_performance = selection_performance_new
        else:
            metrics.update(args.top_metric)
    return metrics
