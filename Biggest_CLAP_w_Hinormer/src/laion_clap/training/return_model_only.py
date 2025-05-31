import logging
import os
import random
from datetime import datetime
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from clap_module import create_model_and_transforms, trace_model, create_model, get_model_cfg
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate
from clap_module.utils import dataset_split, get_optimizer


def maintain_ckpts(args, startidx, all_idx_len):
    for i in reversed(range(startidx, all_idx_len)):
        if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt")):
            os.rename(
                os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                os.path.join(args.checkpoint_path, f"epoch_top_{i+1}.pt"),
            )
    if os.path.exists(
        os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt")
    ):
        os.remove(os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt"))
    return


def update_top_k_performance(
    new_metrics_inputs, current_top_k_ckpt_metrics, args, ckpt, bignumbetter=True
):
    """
    Record the top-k performance of the current epoch.
    current_top_k_metrics is a dictionary of the form: {1: top_1_ckpt_measure, 2: top_2_ckpt_measure, ...}
    """
    if isinstance(new_metrics_inputs, (list, tuple)):
        new_metrics_inputs = np.mean(new_metrics_inputs)
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
        )
    elif isinstance(new_metrics_inputs, dict):
        new_metrics_inputs = np.mean(list(new_metrics_inputs.values()))
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
        )
    elif isinstance(new_metrics_inputs, (float, int)):
        update_flag = {k: False for k in current_top_k_ckpt_metrics.keys()}
        sorted_keys = sorted(current_top_k_ckpt_metrics.keys())
        sorted_values = sorted(
            current_top_k_ckpt_metrics.values(), reverse=bignumbetter
        )
        sorted_values_ = copy.deepcopy(sorted_values)
        sorted_values.append(new_metrics_inputs)
        sorted_values = sorted(sorted_values, reverse=bignumbetter)
        sorted_values = sorted_values[:-1]

        if sorted_values == sorted_values_:
            return current_top_k_ckpt_metrics, new_metrics_inputs
        else:
            for i in range(len(sorted_keys)):
                if current_top_k_ckpt_metrics[sorted_keys[i]] != sorted_values[i]:
                    current_top_k_ckpt_metrics[sorted_keys[i]] = sorted_values[i]
                    update_flag[sorted_keys[i]] = True
            for i in range(len(update_flag)):
                if update_flag[i]:
                    maintain_ckpts(args, i, len(sorted_keys))
                    torch.save(
                        ckpt,
                        os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                    )
                    break
            return current_top_k_ckpt_metrics, new_metrics_inputs


# def updateifNone(a, b):
#     a = b if None else a
#     return a


def is_pretrained_params(n):
    return (
        n.startswith("transformer")
        or n in ["positional_embedding", "text_projection"]
        or n.startswith("token_embedding")
        or n.startswith("ln_final")
        or n.startswith("logit_scale_t")
    )


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    args = parse_args()
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.amodel = args.amodel.replace("/", "-")
    # download sizes.json file

    # (yusong): the below two lines are for debug
    # print("setting up faulthandler")
    # faulthandler.register(10)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.tmodel == "bert" or args.tmodel == "roberta" or args.tmodel == "bart":
        assert (
            args.pretrained == "" or args.pretrained is None
        ), "bert/roberta/bart text encoder does not support pretrained models."

    # get the name of the experiments
    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                f"model_{args.amodel}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.remotedata and is_master(args):
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
                    os.makedirs(f"./json_files/{dataset_name}/{split}")
                os.system(
                    f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
                )

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(args.logs, args.name, "tensorboard")
            if args.tensorboard
            else ""
        )
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""
        args.checkpoint_path = ""

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ["amp", "fp16", "fp32"]
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use fp32 mixed-precision instead of FP16 and AMP in this model. "
            "They will cause NaN loss and NaN gradients. "
            "FP16 and AMP support needs further verification and tuning, especially for train."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    logging.info(f"openai cache dir: {os.path.expanduser(args.openai_model_cache_dir)}")

    model_cfg_to_load_data = get_model_cfg(args.amodel,
        args.tmodel,
        args.pretrained,
        force_quick_gelu=args.force_quick_gelu,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
    )

    data, g, in_dims, node_cnt = get_data(args, device, model_cfg_to_load_data)
    assert len(data), "At least one train or eval dataset must be specified."
    if args.trace:
        assert "train" not in data, "Cannot train with traced model"

    hinormer_args = {
    "graph": g,
    "in_dims": in_dims,
    "hidden_dim": args.h_hidden_dim,
    "num_layers": args.h_num_layers,
    "num_gnns": args.h_num_gnns,
    "num_heads": args.h_num_heads,
    "dropout": args.h_dropout,
    "temperature": args.h_temperature,
    "num_type": len(node_cnt),
    "beta": args.h_beta
    }

    model, model_cfg = create_model(
        args.amodel,
        args.tmodel,
        args.pretrained,
        hinormer_args,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=True,
        pretrained_audio=args.pretrained_audio,
        pretrained_text=args.pretrained_text,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type
    )

    if args.horovod:
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True, **ddp_args
        )

    print('model_cfg: ',model_cfg)

    return model

def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    trained_model = main()
