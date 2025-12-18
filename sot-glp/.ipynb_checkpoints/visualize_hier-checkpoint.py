from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import math
import argparse

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from clip.clip import _transform
from timm.utils import accuracy

import gallop.lib as lib
import gallop.vlprompt.tools as vlp_tools
import gallop.datasets.tools as dts_tools
from gallop.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from gallop.vlprompt import HierLop
from gallop.vlprompt.tools import GlobalLocalLoss
#torch.autograd.set_detect_anomaly(True)

NoneType = Type[None]

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, Union, List
from PIL import Image

@torch.no_grad()
def _infer_grid_hw_from_tokens(num_tokens: int) -> Tuple[int, int]:
    s = int(num_tokens ** 0.5)
    if s * s != num_tokens:
        # fall back to a skinny rectangle if not square
        # try to find a factorization close to square
        for h in range(int(np.sqrt(num_tokens)), 0, -1):
            if num_tokens % h == 0:
                return h, num_tokens // h
        return num_tokens, 1
    return s, s

def _overlay(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    H, W = img_rgb.shape[:2]
    heat_up = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_color = (255 * plt.cm.jet(heat_up)[:, :, :3]).astype(np.uint8)
    return cv2.addWeighted(img_rgb, 1.0, heat_color, alpha, 0)

@torch.no_grad()
def heatmap_from_local_logits(
    local_logits: torch.Tensor,
    class_idx: int,
    grid_hw: Optional[Tuple[int, int]] = None,
    topk: Optional[int] = 20,
    normalize: bool = True,
) -> np.ndarray:
    """
    local_logits: per-token similarity scores that include the local branch.
      accepted shapes (single image B=1):
        - (1, C, L)  -> class-major
        - (1, L, C)  -> token-major
        - (C, L) or (L, C) if batch dim already squeezed
    class_idx: class to visualize
    grid_hw: (Hgrid, Wgrid). If None, inferred as square-ish.
    """
    x = local_logits
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]  # drop batch

    # Reorder to (L,) for the selected class
    if x.dim() == 2:
        if x.shape[0] == class_idx or x.shape[1] == class_idx:
            pass
        # heuristics: decide which dim is class
        # Prefer shape=(C,L) -> pick row; else (L,C) -> pick column
        if x.shape[0] > x.shape[1]:
            # likely (C, L)
            token_scores = x[class_idx]                    # (L,)
        else:
            # likely (L, C)
            token_scores = x[:, class_idx]                 # (L,)
    else:
        # 3D but without batch (shouldn't happen); fallback
        raise ValueError(f"Unexpected local_logits shape: {local_logits.shape}")

    token_scores = token_scores.float()

    # sparsify like GalLoP (optional)
    if topk is not None and topk > 0 and topk < token_scores.numel():
        vals, idxs = torch.topk(token_scores, k=topk, largest=True)
        mask = torch.zeros_like(token_scores)
        mask[idxs] = vals
        token_scores = mask

    # normalize to [0,1]
    if normalize:
        token_scores = token_scores - token_scores.min()
        mx = token_scores.max()
        if mx > 0:
            token_scores = token_scores / mx

    # grid
    L = token_scores.numel()
    if grid_hw is None:
        Hg, Wg = _infer_grid_hw_from_tokens(L)
    else:
        Hg, Wg = grid_hw
    heat = token_scores.reshape(Hg, Wg).cpu().numpy()
    return heat

@torch.no_grad()
def visualize_from_local_logits(
    pil_image: Image.Image,
    local_logits: torch.Tensor,
    class_names: List[str],
    class_idx: Optional[int] = None,
    gl_probs: Optional[torch.Tensor] = None,  # if provided, picks top-1 when class_idx is None
    grid_hw: Optional[Tuple[int,int]] = None,
    topk: Optional[int] = 20,
    save_path: Optional[str] = None,
):
    # pick class
    if class_idx is None:
        if gl_probs is None:
            raise ValueError("Provide class_idx or gl_probs to choose top-1 class.")
        class_idx = int(torch.argmax(gl_probs, dim=1).item() if gl_probs.dim()==2 else torch.argmax(gl_probs).item())

    heat = heatmap_from_local_logits(local_logits, class_idx, grid_hw=grid_hw, topk=topk)

    img_rgb = np.array(pil_image.convert("RGB"))
    overlay = _overlay(img_rgb, heat, alpha=0.45)

    title = class_names[class_idx] if isinstance(class_idx, int) else str(class_idx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb);  axes[0].set_title("Image");  axes[0].axis("off")
    axes[1].imshow(overlay);  axes[1].set_title(f"Local-token similarity: {title}");  axes[1].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[viz] saved to {save_path}")
    else:
        plt.show()


def train_one_epoch(
    model: HierLop,
    train_loader: DataLoader,
    loss_fn: GlobalLocalLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
) -> lib.DictAverage:
    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")

    class_names = train_loader.dataset.all_names
    #args.use_fp16 = torch.bool(args.use_fp16)
    if not args.learn_global_prompt and not args.learn_local_prompts:
        with torch.no_grad(), autocast("cuda",torch.float16):
            text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = local_text_features = None

    model.train()
    optimizer.zero_grad()
    track_loader = lib.track(train_loader, f"Epoch {epoch} / {args.max_epoch}")
    for i, batch in enumerate(track_loader):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        with autocast("cuda",torch.float16):
            global_logits, local_logits = model(images, class_names, text_features, local_text_features)
            
            loss,global_loss,local_loss = loss_fn(global_logits, local_logits, targets, model.logit_scale.exp())

        fp16_scaler.scale(loss).backward()
        track_loader.set_postfix({"gpu": torch.cuda.max_memory_allocated() / 1024**3})
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        optimizer.zero_grad()

        gl_probs, global_probs, local_probs = model.create_prediction_scores_last(global_logits, local_logits)
        #gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        topk = accuracy(gl_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        meter.update(
            {
                "loss": loss.detach().item(),
                "global_loss": global_loss.detach().item(),
                "local_loss": local_loss.detach().item(),
                "top1": topk[0],
                "top1_global": global_topk[0],
            },
            images.size(0),
        )

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_local": local_topk[0],
                },
                images.size(0),
            )

    progress.display_summary()

    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    model: HierLop,
    val_loader: DataLoader,
    class_names,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names_original = val_loader.dataset.all_names

    with autocast("cuda",torch.float16):
        text_features, local_text_features = model.encode_text(class_names)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
        same_order = (class_names_original == class_names)
        if not same_order:

            name2idx = {name: i for i, name in enumerate(class_names)}
            idx = np.fromiter((name2idx[name] for name in class_names_original),
                      dtype=np.int64)
            local_text_features = local_text_features[idx , ...]
            text_features = text_features[idx , ...]
                

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]
    for batch in lib.track(val_loader, f"Evaluating on {dataset_name}"):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast("cuda",torch.float16):
            global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)

            if return_scores:
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        gl_probs, global_probs, local_probs = model.create_prediction_scores_last(global_logits, local_logits)
        #gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))

            topk = accuracy(gl_probs, targets, topk=(1,))

            logs = {
                "top1": topk[0],
                "top1_global": global_topk[0],
                "top1_local": local_topk[0],
            }
        else:
            logs = {
                "top1": global_topk[0],
                "top1_global": global_topk[0],
            }

        meter.update(logs, images.size(0))

    model.train(mode)
    return meter, test_scores
from typing import Optional, Literal, Union

Array = Union[np.ndarray, torch.Tensor]

def _to_numpy_uint8(img: Array) -> np.ndarray:
    """Accepts (H,W,3) or (3,H,W), values in [0,1] or [0,255]; returns (H,W,3) uint8 RGB."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1,3) and img.shape[0] != img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))  # (3,H,W) -> (H,W,3)
    img = img.astype(np.float32)
    # normalize to [0,255]
    #if img.max() <= 1.0:
    #    img = img * 255.0
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    return x

def _overlay_heat(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """heat in [0,1], (Hg,Wg) → resize to image and overlay using jet colormap."""
    H, W = img_rgb.shape[:2]
    heat_up = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_color = (255 * plt.cm.jet(heat_up)[:, :, :3]).astype(np.uint8)  # RGB
    out = cv2.addWeighted(img_rgb, 1.0, heat_color, alpha, 0)
    return out

def visualize_local_prompts_for_class(
    img: Array,                         # normalized image in [0,1], (H,W,3) or (3,H,W)
    local_logits: Array,                # (196, 1000, 4)
    class_idx: int,                     # class to visualize
    combine: Literal["max", "mean"] = "max",
    topk: Optional[int] = None,         # e.g., 20 for sparsification; None disables
    save_path: Optional[str] = None,
    fig_title: Optional[str] = None,
):
    """
    Builds 6-panels: original, combined heatmap, and four per-prompt heatmaps.
    """
    # --- prep image
    img_rgb = _to_numpy_uint8(img)

    # --- to numpy
    if isinstance(local_logits, torch.Tensor):
        local_logits_np = local_logits.detach().cpu().numpy()
    else:
        local_logits_np = np.asarray(local_logits)

    assert local_logits_np.ndim == 3 and local_logits_np.shape[0] == 196, \
        f"Expected local_logits shape (196, C, 4), got {local_logits_np.shape}"

    # slice: (196, 4) for the chosen class
    token_prompt = local_logits_np[:, class_idx, :]  # (196, 4)
    print(token_prompt[...,0])
    print(token_prompt[...,1])
    # reshape tokens to 14x14 grids per prompt
    Hg = Wg = int(np.sqrt(token_prompt.shape[0]))  # 14
    prompt_maps = token_prompt.reshape(Hg, Wg, 4)  # (14,14,4)

    # sparsify (optional) per prompt
    if topk is not None and topk > 0 and topk < Hg * Wg:
        flat = token_prompt.copy()  # (196,4)
        # keep top-k per prompt, zero the rest
        for p in range(flat.shape[1]):
            idx = np.argpartition(flat[:, p], -topk)[-topk:]
            mask = np.zeros_like(flat[:, p], dtype=np.float32)
            mask[idx] = flat[idx, p]
            flat[:, p] = mask
        prompt_maps = flat.reshape(Hg, Wg, 4)

    # normalize each prompt map to [0,1]
    prompt_maps = np.stack([_normalize01(prompt_maps[..., i]) for i in range(4)], axis=-1)  # (14,14,4)

    # combined map
    if combine == "max":
        combined = np.max(prompt_maps, axis=-1)  # (14,14)
    else:
        combined = np.mean(prompt_maps, axis=-1) # (14,14)
    combined = _normalize01(combined)

    # overlays

    overlay_combined = _overlay_heat(img_rgb, combined, alpha=0.45)
    overlays_per_prompt = [_overlay_heat(img_rgb, prompt_maps[..., i], alpha=0.45) for i in range(4)]

    # --- plot
    plt.figure(figsize=(12, 8))
    plt.suptitle(fig_title or f"Local prompt visualization — class {class_idx} (combine={combine}, topk={topk})", y=0.98)

    ax1 = plt.subplot(2, 3, 1); ax1.imshow(img_rgb); ax1.set_title("Image"); ax1.axis("off")
    ax2 = plt.subplot(2, 3, 2); ax2.imshow(overlay_combined); ax2.set_title("Combined heatmap"); ax2.axis("off")

    for i in range(4):
        ax = plt.subplot(2, 3, 3 + i)
        ax.imshow(overlays_per_prompt[i])
        ax.set_title(f"Prompt {i}")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] saved to {save_path}")
    else:
        plt.show()


@torch.no_grad()
def visualize(
    model: HierLop,
    val_loader: DataLoader,
    class_names,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names_original = val_loader.dataset.all_names

    with autocast("cuda",torch.float16):
        text_features, local_text_features = model.encode_text(class_names)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
        same_order = (class_names_original == class_names)
        if not same_order:

            name2idx = {name: i for i, name in enumerate(class_names)}
            idx = np.fromiter((name2idx[name] for name in class_names_original),
                      dtype=np.int64)
            local_text_features = local_text_features[idx , ...]
            text_features = text_features[idx , ...]
                

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]
    for batch in lib.track(val_loader, f"Evaluating on {dataset_name}"):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast("cuda",torch.float16):
            global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
        print("target",targets[1])
        
        visualize_local_prompts_for_class(
            img=images[2], 
            local_logits=local_logits[2], 
            class_idx=0,       # you provide this
            combine="mean",                  # or "mean"
            topk=2,                        # None to disable sparsification
            save_path="local_prompts_viz.png"
        )
        print(a)


@torch.no_grad()
def evaluate_ood(
    model: HierLop,
    val_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    test_scores: Optional[np.ndarray] = None,
) -> lib.DictAverage:
    metrics = defaultdict(dict)

    class_names = val_loader.dataset.all_names

    with autocast("cuda",torch.float16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    if test_scores is None:
        test_scores = np.zeros(len(val_loader.dataset))
        for batch in lib.track(val_loader, "Computing ood scores for Test"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast("cuda",torch.float16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = np.zeros(len(ood_loader.dataset))
        for batch in lib.track(ood_loader, f"Computing ood scores for {ood_name}"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast("cuda",torch.float16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                ood_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        metrics[ood_name]["fpr95"] = lib.get_fpr(test_scores, ood_scores)
        metrics[ood_name]["auroc"] = lib.get_auroc(test_scores, ood_scores)

    model.train(mode)
    return metrics


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser("Learning prompts for CLIP with local and global features")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="/share/DEEPLEARNING/datasets", type=str)
    parser.add_argument("--save_dir", default="./results/", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_shots", default=16, type=int, help="Number of shots by class. -1 means the whole dataset")
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--topk", default=[5, 10, 15, 20], type=int, nargs="+")
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompt", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--ood_method", default="GL-MCM", type=str)
    parser.add_argument("--ood_temp_scale", default=1000.0, type=float)

    parser.add_argument("--clip_name", required=True, choices=clip_model_names, type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    args = parser.parse_args()

    lib.setup_logger(args.exp_name)
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)

    args.eval_domains = args.eval_domains and (args.dataset_name == "imagenet")
    args.eval_ood = args.eval_ood and (args.dataset_name == "imagenet")

    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up Imagenet dataset train
    train_dataset, val_dataset, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform)
    template = "A photo of a {}" if (args.learn_global_prompt or args.learn_local_prompts) else template

    train_dataset = dts_tools.create_few_shots_dataset(train_dataset, args.num_shots, seed=args.seed)
    lib.LOGGER.info("Using template: " + template.format("<class_name>"))

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(val_dataset, batch_size=args.inference_batch_size)

    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)

    if args.eval_domains:
        domains_loaders = return_domains_loaders(args.data_dir, val_transform)

    # Setting-up model
    model = HierLop(
        clip_name=args.clip_name,
        use_local_features=args.use_local_features,
        checkpointing_segments=args.checkpointing_segments,
        template=template,
        learn_local_proj=args.learn_local_proj,
        learn_local_prompts=args.learn_local_prompts,
        learn_global_prompt=args.learn_global_prompt,
        class_names=train_dataset.all_names,
        n_global_prompts=args.n_global_prompts,
        n_local_prompts=args.n_local_prompts,
        prompts_batch_size=args.prompts_batch_size,
        ood_method=args.ood_method,
        ood_temp_scale=args.ood_temp_scale,
        topk=args.topk,
        parallel_text_encoder=args.parallel_text_encoder,
        parallel_vision_encoder=args.parallel_vision_encoder,
    )

    model.initialize_prompt()

    # eventually load pre-trained prompts

    ckpt = torch.load(args.checkpoint_path,weights_only=False)
    print(ckpt["state_dict"]["local_prompts"][0,0])
    print(ckpt["state_dict"]["local_prompts"][0,1])

    lib.load_checkpoint(model, args.checkpoint_path)
    print("Freezed Clip")
    model.v2v_use()
    model.freeze_clip()

    #model.apply_lora()
    model = model.cuda()
    print(model.local_prompts[0,0])
    print(model.local_prompts[0,1])
    #print(a)

    # setting-up loss
    loss_fn = GlobalLocalLoss(
        use_global_loss=args.use_global_loss,
        use_local_loss=args.use_local_loss,
        topk=args.topk,
        global_dropout_p=args.global_dropout_p,
    )

    # Setting-up optimizer
    optimizer = vlp_tools.get_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)

    # Setting-up scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    # Setting-up GradScaler for amp
    fp16_scaler = GradScaler("cuda",enabled=args.use_fp16)

    # Training loop
    for epoch in range(args.max_epoch):
        if not args.eval_only:
            assert args.use_local_loss or args.use_global_loss or args.learn_local_prompts or args.learn_global_prompt, "At least one of use_local_loss or use_global_loss or learn_local_prompts or learn_global_prompt must be True"
            train_meter = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
            )

            lib.save_checkpoint(args.save_dir, epoch, model, optimizer, lr_scheduler, fp16_scaler, train_meter, args)

        if ((epoch % args.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == args.max_epoch) or args.eval_only:
            lib.LOGGER.info("Evaluation")
            val_meter, test_scores = visualize(model, val_loader, train_loader.dataset.all_names, args, return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)))
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))

            if args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)):
                ood_metrics = evaluate_ood(model, val_loader, ood_loaders, args, test_scores=test_scores)
                lib.LOGGER.info(f"OOD Evaluation metrics with temperature scale {args.ood_temp_scale} (FPR95 / AUROC): ")
                lib.log_ood_metrics(ood_metrics)

            if args.eval_domains and (args.eval_only or (epoch + 1 == args.max_epoch)):
                metrics = {}
                for domain_name, domain_loader in domains_loaders.items():
                    metrics[domain_name], _ = visualize(model, domain_loader, args)
                    lib.LOGGER.info(f"Evaluation metrics for {domain_name}: " + " ".join([" *"] + metrics[domain_name].summary()))
                avg_top1 = np.mean([metrics[domain_name].avg["top1"] for domain_name in domains_loaders.keys()])
                lib.LOGGER.info(f"Average evaluation metrics for domains: * top1: {avg_top1: .3f}")

            if args.eval_only:
                break
