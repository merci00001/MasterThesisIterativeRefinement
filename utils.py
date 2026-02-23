import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from datasets import load_dataset
import re

from scipy.ndimage import label
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.stats import multivariate_normal
from torchvision.utils import save_image
from torchvision import transforms
from skimage.filters import threshold_otsu, threshold_multiotsu

from scipy.spatial.distance import cdist
import cv2


def bbox_from_mask(mask) :
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return [x0, y0, width, height]





def elbow_chord(values):
    # Returns threshold value (y), not index
    if len(values) <= 2:
        return min(values) if values else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)  # ascending
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return y[0]
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def binarize_mean_relu(M, ent = None, do_max = False):
    d = 1.0
    

    m = M.mean() * d
    if do_max:
        m = M.max() * 0.5

    B = np.maximum(M - m, 0.0)
    return (B > 0).astype(np.uint8)


def spatial_entropy(attn_map_2d: torch.Tensor, threshold: float):
    # attn_map_2d: [P, P]
    S = attn_map_2d
    mean_val = torch.mean(S)

    B = torch.relu(S - mean_val*2) 
    B_np = B.detach().cpu().to(torch.float32).numpy()
    binary = (B_np > threshold).astype(np.int32)

    from scipy.ndimage import label
    labeled, num = label(binary, structure=np.ones((3, 3)))

    total = float(B.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}

    # Probability mass per component
    probs = []
    for i in range(1, num + 1):
        comp_sum = B_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}

def analyze_heads(attn: torch.Tensor, W = 24, H1 = 24):
    """Analyze heads and return a ranked list.

    attn: [L, H, 1, V]
    meta: includes patch_size (P)
    """
    L, H, _, V = attn.shape
    P = 24 #int(meta.get("patch_size", int(np.sqrt(V))))

    # Criterion 1: head sums over image patches
    sums = []
    for l in range(L):
        for h in range(H):
            s = float(attn[l, h, 0].sum().item())
            sums.append(s)

    thr_val = elbow_chord(sums) #if cfg.logic.threshold.method == "chord" else min(sums)

    # Analyze Criterion 2 only for heads above thr_val (by value)
    results =  []
    idx = 0
    for l in range(L):
        for h in range(H):
            s = sums[idx]
            idx += 1
            if s < thr_val:
                se = float("inf")
                bottom_row_focus = False
                n_comp = 0
            else:
                a2d = attn[l, h, 0].reshape((W, H1))
                se_res = spatial_entropy(a2d, 0.001)
                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > 0.05).any())
                se = float(se_res["spatial_entropy"])    # lower is better
                labeled = se_res["labeled_array"]
                n_comp = int(se_res["num_components"])
            results.append({
                "layer": l,
                "head": h,
                "attn_sum": s,
                "spatial_entropy": se,
                "bottom_row_focus": bottom_row_focus,
                "num_components": n_comp,
            })

    # Filter and sort: keep heads above threshold, prefer non-bottom-row
    kept = [r for r in results if np.isfinite(r["spatial_entropy"]) and r["attn_sum"] >= thr_val and not r["bottom_row_focus"] and r["layer"] > 1]
    if len(kept) < 1:
        # fallback: take top by sum if too few
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: 1]

    kept.sort(key=lambda x: x["spatial_entropy"])  # ascending
    return kept

def combine_heads(attn: torch.Tensor, selected, W,H, sigma):
    """Combine selected heads with optional Gaussian smoothing.

    attn: [L, H, 1, V]
    Returns: combined 2D map [P, P] as numpy float32
    """
    M = np.zeros((W, H), dtype=np.float32)
    ent = 0
    for item in selected:
        l, h = item["layer"], item["head"]
        ent += item["spatial_entropy"]
        a2d = attn[l, h, 0].reshape(W, H).detach().cpu().to(torch.float32).numpy()
        if sigma and sigma > 0:
            a2d =  gaussian_filter(a2d, sigma=sigma) #gaussian_filter(a2d, sigma=sigma)uniform_filter
        M += a2d.astype(np.float32)
 
    return M, ent

def plot_mask(img, top_indices, vis_len, question_id, question):
    P = int(np.sqrt(vis_len))
    
    mask = torch.zeros(vis_len, dtype=torch.bool, device = "cuda")
    mask[top_indices] = True
    
    mask = mask.reshape((P,P))

    img_np = np.array(img)  # Convert to NumPy array
    img_h, img_w = img_np.shape[:2]
    mask_h, mask_w = mask.shape

    mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to 0–255 for PIL

    # Create PIL image and resize
    mask_pil = Image.fromarray(mask_np)
    resized_mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), resample=Image.NEAREST)


    resized_mask = np.array(resized_mask_pil) > 127  # Convert back to boolean

    # Apply the mask
    masked_img = img_np.copy()
    masked_img[~resized_mask] = 0

    masked_pil = Image.fromarray(masked_img)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Masked image
    axes[1].imshow(masked_pil)
    axes[1].set_title("Masked Image")
    axes[1].axis("off")

    plt.suptitle(f"Question : {question}", fontsize=16)

    # Save and close
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
    plt.savefig(f"/cluster/scratch/mgroepl/debug/{question_id}Mask.jpg", dpi=300)
    plt.close()




def save_tensor_image(t, path):
    pil_image = Image.fromarray(np_array)

    # Save the image
    pil_image.save(path)





def find_crop_in_global(global_img: Image.Image, crop_img: Image.Image):
    """
    Find the pixel bbox of crop_img inside global_img using template matching.
    Returns (x_min, y_min, x_max, y_max) in global pixel coordinates.
    """
    global_np = np.array(global_img.convert("RGB"))
    crop_np   = np.array(crop_img.convert("RGB"))

    # Convert to grayscale for matching
    global_gray = cv2.cvtColor(global_np, cv2.COLOR_RGB2GRAY)
    crop_gray   = cv2.cvtColor(crop_np,   cv2.COLOR_RGB2GRAY)

    # Resize crop to match the scale it would appear at in the global image
    # (necessary because the crop was upscaled by PIL when cropping)
    h_g, w_g = global_gray.shape
    h_c, w_c = crop_gray.shape

    if h_c > h_g or w_c > w_g:
        # Crop is bigger than global after resampling — scale it down
        scale = min(h_g / h_c, w_g / w_c)
        new_w = int(w_c * scale)
        new_h = int(h_c * scale)
        crop_gray = cv2.resize(crop_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h_c, w_c = new_h, new_w

    result = cv2.matchTemplate(global_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(result)

    x_min, y_min = top_left
    x_max = x_min + w_c
    y_max = y_min + h_c

    return (x_min, y_min, x_max, y_max), score



def gini(grad_map: np.ndarray) -> float:
    flat = grad_map.flatten().astype(np.float64)
    flat = flat - flat.min()  # shift to non-negative
    flat = np.sort(flat)
    n = len(flat)
    total = flat.sum()
    if total == 0:
        return 0.0
    # Standard formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    indices = np.arange(1, n + 1)  # 1-indexed
    return float((2 * (indices * flat).sum()) / (n * total) - (n + 1) / n)


def adaptive_sigma_std(grad_map: np.ndarray, scale=0.5) -> float:
    """Sigma proportional to relative spread of gradient values."""
    return float(np.std(grad_map) / (grad_map.max() + 1e-8) * grad_map.shape[0] * scale)

def adaptive_sigma_gini(grad_map: np.ndarray, min_s=0.5, max_s=5.0) -> float:
    """More concentrated map (high Gini) → less blurring needed."""
    g = gini(grad_map)
    return float(min_s + (1.0 - g) * (max_s - min_s))

def adaptive_sigma_resolution(grad_map: np.ndarray, blob_fraction=0.25) -> float:
    """Sigma tied to expected blob size in the grid."""
    h, w = grad_map.shape
    blob_radius = np.sqrt(blob_fraction * h * w / np.pi)
    return float(blob_radius * 0.3)





def bbox_from_att_image_adaptive(att_map: np.ndarray, image_size: tuple,
                                 bbox_size: int = 336) -> tuple:
    """
    Find the bbox of the most salient region in a 2D attention map.

    Args:
        att_map:    2D numpy array, e.g. (24, 24) or (48, 48).
        image_size: (width, height) of the target image in pixels.
        bbox_size:  Base crop size in pixels (default 336).

    Returns:
        (x1, y1, x2, y2) pixel coordinates clipped to image bounds.
    """
    ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    block_size = (image_size[0] / att_map.shape[1],
                  image_size[1] / att_map.shape[0])

    best_diff, best_pos, best_block, best_ratio = -np.inf, (0, 0), (1, 1), 1.0

    for ratio in ratios:
        bw = min(int(bbox_size * ratio / block_size[0]), att_map.shape[1])
        bh = min(int(bbox_size * ratio / block_size[1]), att_map.shape[0])

        if att_map.shape[1] - bw < 1 and att_map.shape[0] - bh < 1:
            if ratio == 1.0:
                return 0, 0, image_size[0], image_size[1]
            continue

        # Integral image for O(1) rectangle sums
        integral = np.cumsum(np.cumsum(att_map, axis=0), axis=1)
        out_h, out_w = att_map.shape[0] - bh + 1, att_map.shape[1] - bw + 1
        y_idx, x_idx = np.mgrid[0:out_h, 0:out_w]
        y2, x2 = y_idx + bh - 1, x_idx + bw - 1

        sliding_att  = integral[y2, x2]
        sliding_att -= np.where(y_idx > 0, integral[y_idx - 1, x2], 0)
        sliding_att -= np.where(x_idx > 0, integral[y2, x_idx - 1], 0)
        sliding_att += np.where((y_idx > 0) & (x_idx > 0), integral[y_idx - 1, x_idx - 1], 0)

        flat_idx  = np.argmax(sliding_att)
        iy, ix    = np.unravel_index(flat_idx, sliding_att.shape)
        max_att   = float(sliding_att[iy, ix])
        max_pos   = (ix, iy)  # (x, y)

        neighbours = []
        if ix > 0:              neighbours.append(sliding_att[iy, ix - 1])
        if ix < out_w - 1:     neighbours.append(sliding_att[iy, ix + 1])
        if iy > 0:              neighbours.append(sliding_att[iy - 1, ix])
        if iy < out_h - 1:     neighbours.append(sliding_att[iy + 1, ix])
        diff = (max_att - np.mean(neighbours)) / (bw * bh) if neighbours else 0.0

        if diff > best_diff:
            best_diff, best_pos, best_block, best_ratio = diff, max_pos, (bw, bh), ratio

    half = bbox_size * best_ratio / 2
    x_center = int(best_pos[0] * block_size[0] + block_size[0] * best_block[0] / 2)
    y_center = int(best_pos[1] * block_size[1] + block_size[1] * best_block[1] / 2)
    x_center = int(np.clip(x_center, half, image_size[0] - half))
    y_center = int(np.clip(y_center, half, image_size[1] - half))

    return (
        max(0, int(x_center - half)),
        max(0, int(y_center - half)),
        min(image_size[0], int(x_center + half)),
        min(image_size[1], int(y_center + half)),
    )


import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# High-pass filter
# ---------------------------------------------------------------------------

def high_pass_filter(image: Image.Image, target_size: int, reduce: bool = False) -> np.ndarray:
    """
    Apply a high-pass filter to a PIL image by subtracting a Gaussian-blurred version.

    Args:
        image:       Input PIL image.
        target_size: Resize image to (target_size, target_size) before filtering.
        reduce:      If True, return a spatially reduced map.

    Returns:
        2D numpy array of high-frequency content.
    """
    img = image.resize((target_size, target_size)).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    blurred = gaussian_filter(img_np, sigma=3.0)
    hp = np.abs(img_np - blurred)
    return hp


# ---------------------------------------------------------------------------
# Core gradient function
# ---------------------------------------------------------------------------

def pure_gradient_llava(image, image_tensor, image_tensor_general,
                        image_sizes, input_ids, input_ids_general,
                        model, patch_size=14, image_resolution=336,
                        layer=-1, mode="entropy"):

    grad_specific = calc_grad_image(model, image_tensor, input_ids,
                                    image_sizes, layer=layer, mode=mode)
    grad_general  = calc_grad_image(model, image_tensor_general, input_ids_general,
                                    image_sizes, layer=layer, mode=mode)

    # Relative gradient
    eps = np.percentile(grad_general, 10) + 1e-8
    relative = grad_specific / (grad_general + eps)

    # High-pass filter
    hp = high_pass_filter(image, target_size=image_resolution)
    hp_binary = hp > np.median(hp)
    if hp_binary.shape != relative.shape:
        from PIL import Image as PILImage
        hp_pil = PILImage.fromarray(hp_binary.astype(np.uint8) * 255).resize(
            (relative.shape[1], relative.shape[0]), resample=PILImage.NEAREST
        )
        hp_binary = np.array(hp_pil) > 0

    filtered = relative * hp_binary

    # Block-reduce to patch grid (e.g. 336/14 = 24x24)
    grid_size = image_resolution // patch_size
    block = max(1, filtered.shape[0] // grid_size)
    grad_map = block_reduce(filtered, block_size=(block, block), func=np.mean)
    return grad_map[:grid_size, :grid_size]


def get_adaptive_sigma(grad_map: np.ndarray, mode="std") -> float:
    fns = {
        "std":        adaptive_sigma_std,
        "gini":       adaptive_sigma_gini,
        "resolution": adaptive_sigma_resolution,
    }
    if mode not in fns:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(fns)}")
    sigma = fns[mode](grad_map)
    return float(np.clip(sigma, 0.5, 5.0))

def get_disjoint_segments(attn_layers, W,H,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None, grad = None, plot = False, el = None):
    ent  = 0
    filtered_mask = None
   
    if insert_mask is None and grad is None:
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]

        selected = analyze_heads( attn_last_to_vis.detach().cpu(), W = W, H1 = H)
    
        combo_raw, ent = combine_heads(attn_last_to_vis, selected[: 15], W=W,H=H, sigma=0.0)
        #combo_raw = combo_raw.flatten()
        combo, ent = combine_heads(attn_last_to_vis, selected[: 10], W=W,H=H, sigma=2.0)

        if plot:
        
            path = get_unique_filename("/cluster/scratch/mgroepl/debug/test/", "attCombined.png")
            # Save the image
            plt.imsave(path, combo_raw)


        max_value = combo_raw.max()

        # Get the indices where the values are greater than max * 0.9
        indices = np.where(combo_raw > max_value * 0.5)
        #combo = uniform_filter(combo, size=3) 
        
        
        mask_grid = binarize_mean_relu(combo, ent)

    elif grad is not None:
        grad_orig = grad.reshape(W,H).detach().cpu().to(torch.float32).numpy()
        #grad_orig[grad_orig <grad_orig.mean()] = 0.0




        



        temperature = 0.1
        grad_orig_flat2 = grad.clone()
        #grad_orig_flat2 = F.softmax(grad_orig_flat2 / temperature, dim=-1)
        grad_orig = grad_orig_flat2.reshape(W,H).detach().cpu().to(torch.float32).numpy()


        top_percentile = 99
        high_thresh = grad_orig.max()*0.01

        # Binary mask of high activations
        sigma = 1.5 #adaptive_sigma_gini(grad_orig, max_s = 2.5)



        grad_orig2 =gaussian_filter(grad_orig, sigma=sigma)#  gaussian_filter(grad_orig, sigma=1.5)# gaussian_filter(grad_orig, sigma=1.5)  cv2.bilateralFilter(grad_orig, d=9, sigmaColor=75, sigmaSpace=75)
        





        if el is None:
            el =elbow_chord(grad_orig2.flatten()) # threshold_multiotsu(grad_orig2, classes=3)[-1] #  elbow_chord(grad_orig2.flatten())  # elbow_chord(grad_orig2.flatten()) #  elbow_chord(grad_orig2.flatten())#  elbow_chord(grad_orig2.flatten()) #elbow_chord(grad_orig2.flatten()) #  grad_orig2.max() *0.5  threshold_otsu(grad_orig2)


        ent = spatial_entropy(torch.tensor(grad_orig2), el)
        ent = ent["spatial_entropy"]
        #ent = -gini(grad_orig2)  #ent["spatial_entropy"]
        #ent = spatial_enth(grad_orig)
        print("spatial enthropy ", ent)



        grad_mask = grad_orig2 > el 
        #grad_mask = grad_orig2 > el*min(ent["spatial_entropy"],1) #grad_orig.mean()#*0.3   #grad_orig.max()*0.3  #binarize_mean_relu(grad_orig, do_max = True)
        blob_mask = grad_mask.astype(bool)
        if False: #plot:
            print("plotting...")
            plt.imshow(blob_mask, cmap='viridis')  # 'gray' colormap for grayscale image
            plt.axis('off')  # Turn off axis for a cleaner image
            plt.colorbar()
            # Save the image to a file
            plt.savefig(f'/cluster/scratch/mgroepl/debug/test/blob_mask{ent}.png', bbox_inches='tight', pad_inches=0)



            plt.imshow(grad_orig2, cmap='viridis')  # 'gray' colormap for grayscale image
            plt.axis('off')  # Turn off axis for a cleaner image
            plt.colorbar()
            # Save the image to a file
            plt.savefig(f'/cluster/scratch/mgroepl/debug/test/output_image{ent}.png', bbox_inches='tight', pad_inches=0)

            # Close the plot to free memory
            plt.close()



        mask_grid = blob_mask



        combo = grad_orig
    else:
        mask_grid = insert_mask
    if return_single:
        return [mask_grid], None,None#[bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
   
    sorted_vals_per_segment = [ combo[b]   for b in segment_masks]

    
    return segment_masks,  sorted_vals_per_segment, ent


import numpy as np












def get_bbox_indices(attn_layers, P,begin_pos_vis_att, vis_len = 576 , do_grid = True, returnBBOX = False):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[3: 8], P=P, sigma=2.0)
    mask_grid = binarize_mean_relu(combo, ent)
    if do_grid:
        flattened = torch.flatten(torch.from_numpy(mask_grid))
        indices = torch.where(flattened)[0]
        return indices, ent
    bbox_grid = bbox_from_mask(mask_grid)
    if returnBBOX:
        return bbox_grid, ent


    return box_to_indices(bbox_grid, P), ent

def box_to_indices(bbox, P):
    x,y,w,h = bbox
    x1 = x
    y1 = y
    x2 = x + w -1
    y2 =  y + h -1
    ys, xs = np.meshgrid(np.arange(y1, y2 + 1), np.arange(x1, x2 + 1), indexing='ij')

    # Convert (y, x) to 1D indices in a flattened image
    indices = ys * P + xs
  
    return indices.flatten()




def get_indices_percent(attn_layers, begin_pos_vis_att, vis_len = 576, mode = "selected", topK = 0.9, largest = False, sample = False,attn_mean_all = None, general_att_map = None, width = 1, height = 1,grad = None):

    
    ent = 0
    if mode == "topK":
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

        # Step 2: Squeeze query dimension (it's size 1)
        attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
        attn_mean_all = attn_mean_heads.mean(dim=0) 
        
    elif mode == "selected":
     
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

        # Step 2: Squeeze query dimension (it's size 1)
        attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
        selected = analyze_heads( attn_last_to_vis.detach().cpu())


        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], W=24,H=24, sigma=None)
        
        newAtt = torch.zeros((attn_layers.shape[2],attn_layers.shape[3]), dtype = float, device = "cuda")
     
        for x in selected:
            l = x["layer"]
            h = x["head"]
            newAtt += attn_layers[l,h,:,:]
        
        attn_mean_all = newAtt[-1, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    elif mode == "general":
 

        attn_pic = attn_layers[14,:,0,begin_pos_vis_att:begin_pos_vis_att + vis_len].mean(0).reshape(24,24) / general_att_map
        attn_pic = attn_pic.cpu().numpy()
        attn_pic = gaussian_filter(attn_pic.astype(np.float32), sigma=2.0)
        attn_mean_all = attn_pic.flatten()
    elif mode == "grad":
        attn_mean_all = grad
    top_k = int(topK * attn_mean_all.shape[0]) 
    if sample:
        attn_scores = attn_mean_all.clone()  # don't modify the original
        attn_scores = attn_scores - attn_scores.min()  # optional: make all scores non-negative
        prob = attn_scores / attn_scores.sum()
        sampled_indices = torch.multinomial(prob, num_samples=top_k, replacement=False)
        return sampled_indices, ent
    if False:
        indeces = torch.arange(attn_mean_all.shape[0]).reshape((24,24)).cuda()
        attn_mean_all_reshaped = attn_mean_all.reshape((24,24))
        w_sum = attn_mean_all_reshaped.sum(dim = 0)

        top_values, top_indices = torch.topk(w_sum, k=width, largest = largest)
       
        total_ind = []
        for x in top_indices:
            h_val = attn_mean_all_reshaped[x,:]
            top_values, top_indices = torch.topk(h_val, k=height, largest = largest)
            total_ind.append(indeces[x,top_indices])
        total_ind = torch.stack(total_ind).flatten()
        return total_ind
    else:
        top_values, top_indices = torch.topk(attn_mean_all, k=top_k, largest = largest)
        return top_indices, ent





def build_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    """
    Re-creates the combined causal + padding mask used inside LLaMA/LLaVA.
    attention_mask: [B, T] (1 for valid tokens, 0 for pad)
    input_shape: (B, T, H) or (B, T)
    inputs_embeds: [B, T, H]
    Returns: [B, 1, T, T] mask with 0 for keep, -inf for mask
    """
    if len(input_shape) == 3:
        bsz, tgt_len, _ = input_shape
    else:
        bsz, tgt_len = input_shape

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # 1. Causal mask (upper-triangular)
    causal_mask = torch.full(
        (tgt_len, tgt_len),
        fill_value=float("-inf"),
        device=device,
        dtype=dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    # 2. Combine with padding mask
    if attention_mask is not None:
        padding_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(dtype).min
        combined_mask = causal_mask + padding_mask
    else:
        combined_mask = causal_mask

    return combined_mask

def project_embed(model, embeds):
    

    projector = model.get_model().mm_projector  
    embeds = projector(embeds.half())
    return embeds




def get_clip_embed(model,image_tensor):

    image_features = model.get_model().get_vision_tower()([image_tensor.squeeze(0).squeeze(0)])
    return image_features[0][0]
   


def get_embedding(model,input_ids,image_tensor,image_sizes, new_pos = None,orig_pos=None):

   
    (
        _,
        position_ids,
        att_mask,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        image_sizes = image_sizes,
        #new_pos = new_pos,
        #orig_pos = orig_pos
    )

    return inputs_embeds, att_mask, position_ids



import torch
import torch.nn.functional as F





def _get_topp_indices(probs, p=0.5):
    """Return indices of the top-p nucleus (smallest set whose cumsum >= p)."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep all tokens up to and including the one that pushes cumsum over p
    cutoff = (cumsum < p).sum().item() + 1
    return sorted_indices[:cutoff]


def _run_full_forward(model, input_embeds, attention_mask):
    """Full model forward pass, returns logits."""
    out = model(
        attention_mask=attention_mask,
        inputs_embeds=input_embeds,
        image_sizes=None,
        output_attentions=False,
        return_dict=True,
    )
    return out.logits


def _run_partial_forward(model, input_embeds, attention_mask, num_layers):
    """Run only the first `num_layers` transformer layers, then lm_head."""
    position_ids = torch.arange(
        input_embeds.shape[1], dtype=torch.long, device=input_embeds.device
    ).unsqueeze(0)

    causal_mask = build_decoder_attention_mask(attention_mask, input_embeds.size(), input_embeds)

    hidden = input_embeds
    for i, layer in enumerate(model.model.layers[:num_layers]):
        layer_out = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
            past_key_value=None,
        )
        hidden = layer_out[0]
      
    return model.lm_head(hidden)


def calc_grad_image(model, image_tensor, input_ids, image_sizes,
                    attention_mask=None, layer=-1, mode="entropy"):
    """
    Like calc_grad but differentiates w.r.t. raw image pixels instead of input_embeds.
    Returns a (H, W) numpy gradient norm map.
    """
    pix = image_tensor.detach().clone().squeeze(1).to(torch.float16).requires_grad_(True)
    pix.retain_grad()

    # Keep vision tower in the graph
    model.model.vision_tower.requires_grad_(True)
    model.model.mm_projector.requires_grad_(True)

    try:
        with torch.set_grad_enabled(True):
            # Build input_embeds — vision tower is grad-enabled so graph stays alive
            (_, position_ids, att_mask, _, input_embeds, _) = (
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None,
                    pix,
                    image_sizes=image_sizes,
                )
            )
            print("input_embeds.requires_grad:", input_embeds.requires_grad)

            # Run the same objective as calc_grad
            if layer == -1:
                logits = _run_full_forward(model, input_embeds, att_mask)
            else:
                logits = _run_partial_forward(model, input_embeds, att_mask, num_layers=layer)

            probs = F.softmax(logits[0, -1], dim=-1)
            probs_nz = probs[probs > 0]

            if mode == "entropy":
                objective = (probs_nz * torch.log(probs_nz)).sum()
            elif mode == "max":
                objective = torch.log(probs_nz.max())
            else:
                raise ValueError(f"Unknown mode: {mode}")

            objective.backward()

    finally:
        model.model.vision_tower.requires_grad_(False)
        model.model.mm_projector.requires_grad_(False)
    print("input_embeds",input_embeds.grad)
    assert pix.grad is not None, "Gradient did not reach image tensor"
    grad = pix.grad.to(torch.float32).detach().cpu().numpy().squeeze()
    if grad.ndim == 4:
        grad = grad.mean(axis=0)
    # (C, H, W) → L2 norm over channels → (H, W)
    return np.linalg.norm(grad.transpose(1, 2, 0), axis=2)

def calc_grad(model, input_embeds, attention_mask=None, layer=-1, mode="entropy", gen_steps=1):
    model.requires_grad_(False)
    input_embeds = input_embeds.detach().clone().requires_grad_(True)
    embedding_table = model.get_input_embeddings().weight
    eos_token_id = model.config.eos_token_id

    with torch.set_grad_enabled(True):
        current_embeds = input_embeds
        current_mask = attention_mask
        prev_embeds = input_embeds        # context before the current step
        prev_mask = attention_mask        # mask before the current step
        actual_steps = gen_steps

        for step in range(gen_steps - 1):
            with torch.no_grad():
                if layer == -1:
                    logits = _run_full_forward(model, current_embeds, current_mask)
                else:
                    logits = _run_partial_forward(model, current_embeds, current_mask, num_layers=layer)

                next_token_id = logits[0, -1].argmax(dim=-1)

                if next_token_id.item() == eos_token_id:
                    print(f"EOS hit at step {step + 1}, backpropagating step {step} instead.")
                    actual_steps = step
                    # Re-run previous context's forward pass with grad enabled
                    prev_input_embeds_final = torch.cat([
                        input_embeds,
                        prev_embeds[:, input_embeds.shape[1]:, :].detach()
                    ], dim=1)
                    if layer == -1:
                        logits = _run_full_forward(model, prev_input_embeds_final, prev_mask)
                    else:
                        logits = _run_partial_forward(model, prev_input_embeds_final, prev_mask, num_layers=layer)
                    probs = F.softmax(logits[0, -1], dim=-1)
                    break

                next_embed = embedding_table[next_token_id].detach()
                next_embed = next_embed.unsqueeze(0).unsqueeze(0)

            # Save previous context before extending
            prev_embeds = current_embeds
            prev_mask = current_mask

            current_embeds = torch.cat([current_embeds.detach(), next_embed], dim=1)
            if current_mask is not None:
                extra = torch.ones(1, 1, dtype=current_mask.dtype, device=current_mask.device)
                current_mask = torch.cat([current_mask, extra], dim=1)

        else:
            # No early EOS — run nth forward pass with grad enabled
            input_embeds_final = torch.cat([
                input_embeds,
                current_embeds[:, input_embeds.shape[1]:, :].detach()
            ], dim=1)
            if layer == -1:
                logits = _run_full_forward(model, input_embeds_final, current_mask)
            else:
                logits = _run_partial_forward(model, input_embeds_final, current_mask, num_layers=layer)
            probs = F.softmax(logits[0, -1], dim=-1)

        # --- Shared objective + backward ---
        probs_nz = probs[probs > 0]
        print("Non-zero values: ", probs_nz.shape)
        topp_indices = _get_topp_indices(probs_nz, p=0.5)

        if mode == "entropy":
            objective = (probs_nz * torch.log(probs_nz)).sum()
        elif mode == "entropyTopP":
            p_nucleus = probs_nz[topp_indices]
            objective = (p_nucleus * torch.log(p_nucleus)).sum()
        elif mode == "max":
            objective = torch.log(probs_nz.max())
        elif mode == "KL":
            uniform = torch.full_like(probs_nz, 1.0 / probs_nz.size(-1))
            objective = (probs_nz * torch.log(probs_nz / uniform)).sum()
        elif mode == "KLTopP":
            p_nucleus = probs_nz[topp_indices]
            uniform_nucleus = torch.full_like(p_nucleus, 1.0 / p_nucleus.size(-1))
            objective = (p_nucleus * torch.log(p_nucleus / uniform_nucleus)).sum()
        else:
            raise ValueError(f"Unknown gradient mode: '{mode}'. "
                             f"Choose from: entropy, entropyTopP, max, KL, KLTopP.")

        objective.backward()
        grads = input_embeds.grad.detach().clone()

    del objective
    torch.cuda.empty_cache()
    print(f"Backpropagated at step {actual_steps}, max prob: {probs_nz.max()}")
    return grads, probs_nz.max().detach()


def resize_box(bbox,image,W= 24, H = 24):
    a,b,width,height = bbox

    W1 = int(image.size[0]/W) 
    H2 = int(image.size[1]/H)

    x_min = int(a * W1)
    x_max = int(x_min + width *W1) 
    y_min = int(b * H2)
    y_max = int(y_min + height *H2)




    box_resized = [x_min, y_min, x_max, y_max]
    return box_resized

def get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = None,num_layer = None, attention_mask = None,position_ids = None ):

    with torch.inference_mode():

        if input_embeds is not None and num_layer is None:
            
            attn_layers = model(
                #input_ids = input_ids,    to(torch.float32)
                attention_mask=attention_mask,
                #images=[image_tensor.to(dtype=torch.float32)],
                inputs_embeds = input_embeds,
                image_sizes=None,
                output_attentions=True,
                return_dict=True,
            
            
            )

        elif num_layer is not None:
            encoder_layers = model.model.layers
            hidden_states = input_embeds
            
            if position_ids == None:
                position_ids = torch.arange(input_embeds.shape[1], dtype=torch.long, device=input_embeds.device).unsqueeze(0)
            attn_outputs = []
            num_layers_to_run = num_layer
            attention_mask = build_decoder_attention_mask(
                attention_mask,
                input_embeds.size(),
                input_embeds
            )
            for i, layer in enumerate(encoder_layers[:num_layers_to_run]):
                # Pass through one layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=True,
                    use_cache=False,
                    past_key_value=None,
                )

                # Get hidden states and attentions
                hidden_states = layer_outputs[0]  # updated hidden state
                if i == 0:
                    first_state = hidden_states
                attn_outputs.append(layer_outputs[1][:,:,-1:,:])
            device = torch.device("cuda:0")  # or whichever GPU you want
            attn_outputs = [x.to(device) for x in attn_outputs]
            attn_outputs = torch.stack(attn_outputs).squeeze(1)
            return attn_outputs, hidden_states

        else:
            attn_layers = model(    
                input_ids=input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                output_attentions=True,
                return_dict=True,
            )
        logits = attn_layers.logits
        last_logits = logits[0, -1]  
        probs = torch.softmax(last_logits, dim=-1)
    
        attn_layers = attn_layers.attentions # tuple length L of [B,H,Tq,Tk]
        attn_layers = torch.stack(attn_layers, dim=0)  # [L, B, H, Tq, Tk]
        attn_layers = attn_layers[:, 0] 
        return attn_layers, probs.max()



def get_unique_filename(folder, filename):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get the base name and extension of the file
    base_name, ext = os.path.splitext(filename)
    
    # Construct the full path
    path = os.path.join(folder, filename)
    
    # Check if the file exists and keep incrementing the number until it's unique
    counter = 0
    while os.path.exists(path):
        counter += 1
        filename = f"{base_name}{counter}{ext}"
        path = os.path.join(folder, filename)
    
    return path



def get_prob_max(model, input_embeds,attention_mask = None):

    attn_layers = model(
        #input_ids = input_ids,    to(torch.float32)
        attention_mask=attention_mask,
        #images=[image_tensor.to(dtype=torch.float32)],
        inputs_embeds = input_embeds,
        image_sizes=None,
        output_attentions=True,
        return_dict=True,   
    )

    logits = attn_layers.logits
    last_logits = logits[0, -1]  
    probs = torch.softmax(last_logits, dim=-1)

    return probs.max()

def plot_arrays(arr1,arr2, name = "plot"):

    plt.figure(figsize=(8, 5))
    plt.scatter(arr1, arr2, c='blue', alpha=0.7)
    plt.xscale('log')  # log scale to spread out small variances
    plt.xlabel('Variance (log scale)')
    plt.ylabel('Size')
    plt.title('Size vs Variance')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(f'/cluster/scratch/mgroepl/res/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Closes the figure and frees memory






def get_image(dataset, index, box = None, P = 24):
    
        line = dataset.questions[index]
        image_file = line["image"]
        qs = line["text"]

    
        image = Image.open(os.path.join(dataset.image_folder, image_file)).convert('RGB')
        if box is not None:
            a,b,width,height = box

            W = image.size[0]/P 
            H = image.size[1]/P
            
            x_min = int(a * W)
            x_max = int(x_min + width *W) 
            y_min = int(b * H)
            y_max = int(y_min + height *H)
            box_resized = (x_min, y_min, x_max, y_max)
            image = image.crop(box_resized)            
        return image






