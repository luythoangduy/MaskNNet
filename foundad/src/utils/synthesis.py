
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import morphology
import cv2


def generate_target_foreground_mask(img: np.ndarray, subclass: str) -> np.ndarray:
    inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    img_tensor = inv_normalize(img)

    img_tensor = torch.clamp(img_tensor, 0, 1)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    img_np_uint8 = (img_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass == 'pill':
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        _, target_background_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_background_mask = (target_background_mask > 0).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['capsules']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        _, target_foreground_mask = cv2.threshold(img_np_uint8[:, :, 2], 100, 255,
                                                    cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pipe_fryum']:
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['bracket_black', 'bracket_brown', 'connector']:
        img_seg = img_np_uint8[:, :, 1]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['bracket_white', 'tubes']:
        img_seg = img_np_uint8[:, :, 2]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = target_background_mask
    elif subclass in ['metal_plate']:
        img_seg = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    else:
        raise NotImplementedError("Unsupported foreground segmentation category")

    target_foreground_mask = morphology.closing(
        target_foreground_mask, morphology.square(6))
    target_foreground_mask = morphology.opening(
        target_foreground_mask, morphology.square(6))

    return target_foreground_mask

class CutPaste(object):
    def __init__(self, colorJitter=0.1):
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter)

    def __call__(self, imgs):
        return imgs, imgs

class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)  # [H, W]


        area = h * w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img 

        valid_indices = []
        for y, x in mask_indices:
            if y + cut_h <= h and x + cut_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img  

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        augmented[:, to_y:to_y+cut_h, to_x:to_x+cut_w] = patch

        return augmented

class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)
    
        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(patch, angle=rot_deg, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

        _, patch_h, patch_w = patch.shape

        to_x = random.randint(0, w - patch_w)
        to_y = random.randint(0, h - patch_h)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img  

        valid_indices = []
        for y, x in mask_indices:
            if y + patch_h <= h and x + patch_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img  

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        mask = torch.ones_like(patch)
        augmented = self.paste_with_mask(augmented, patch, mask, to_y, to_x)

        return augmented

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape

        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top:top+patch_h, left:left+patch_w]
        mask = mask.to(img_region.device)
        img_region = img_region * (1 - mask) + patch * mask
        img[:, top:top+patch_h, left:left+patch_w] = img_region

        return img

class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses):
        batch_size = imgs.shape[0]
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)  # [1, C, H, W]
            subclass = subclasses[i]
            if random.random() < 0.5:
                _, augmented = self.cutpaste_normal(img, subclass)
            else:
                _, augmented = self.cutpaste_scar(img, subclass)
            augmented_imgs[i] = augmented.squeeze(0)

        return imgs, augmented_imgs


class NeuralMaskSynthesizer(object):
    def __init__(
        self,
        area_ratio=(0.02, 0.25),
        aspect_ratio=0.3,
        method="suppress",
        radius=2,
        pca_dim=5,
        channel_topk_ratio=0.1,
        channel_min=32,
        mask_strength=1.0,
    ):
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.method = method
        self.radius = radius
        self.pca_dim = pca_dim
        self.channel_topk_ratio = channel_topk_ratio
        self.channel_min = channel_min
        self.mask_strength = mask_strength

    def __call__(self, imgs: torch.Tensor, feats: torch.Tensor, subclasses):
        batch_size, num_tokens, _ = feats.shape
        grid_size = int(round(math.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected square patch grid, got {num_tokens} tokens")

        synth_feats = feats.clone()
        synth_masks = torch.zeros(batch_size, num_tokens, device=feats.device, dtype=feats.dtype)

        for i in range(batch_size):
            feat_i, mask_i = self.process_sample(imgs[i], feats[i], subclasses[i], grid_size, grid_size)
            synth_feats[i] = feat_i
            synth_masks[i] = mask_i

        return synth_feats, synth_masks

    def process_sample(
        self,
        img: torch.Tensor,
        feat: torch.Tensor,
        subclass: str,
        grid_h: int,
        grid_w: int,
    ):
        target_foreground_mask = generate_target_foreground_mask(img, subclass)
        patch_mask = self.mask_to_patch(target_foreground_mask, grid_h, grid_w).to(feat.device)
        region_mask = self.sample_random_box(patch_mask)

        if region_mask.sum() == 0:
            return feat, torch.zeros(feat.shape[0], device=feat.device, dtype=feat.dtype)

        feat_new = feat.clone()
        token_mask = torch.zeros(feat.shape[0], device=feat.device, dtype=torch.bool)

        region_indices = region_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        for token_idx in region_indices.tolist():
            neigh_idx = self.get_spatial_neighbors(token_idx, grid_h, grid_w, self.radius)
            local_feats = feat[neigh_idx]
            if local_feats.shape[0] <= 1:
                continue

            proj, delta = self.project_to_local_manifold(local_feats, feat[token_idx])
            decisive_dims = self.get_decisive_dims(delta)
            if decisive_dims.numel() == 0:
                continue

            if self.method == "violate":
                delta_norm = delta.norm().clamp_min(1e-6)
                local_scale = local_feats.std(dim=0, unbiased=False).mean().clamp_min(1e-6)
                masked_value = feat[token_idx, decisive_dims]
                masked_value = masked_value + self.mask_strength * local_scale * (delta[decisive_dims] / delta_norm)
            else:
                masked_value = (1.0 - self.mask_strength) * feat[token_idx, decisive_dims]
                masked_value = masked_value + self.mask_strength * proj[decisive_dims]
            feat_new[token_idx, decisive_dims] = masked_value
            token_mask[token_idx] = True

        return feat_new, token_mask.float()

    def sample_random_box(self, foreground_mask: torch.Tensor):
        grid_h, grid_w = foreground_mask.shape
        area = grid_h * grid_w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        box_w = int(round(math.sqrt(target_area * aspect_ratio)))
        box_h = int(round(math.sqrt(target_area / aspect_ratio)))

        box_w = min(max(box_w, 1), grid_w)
        box_h = min(max(box_h, 1), grid_h)

        valid_coords = []
        ys, xs = torch.where(foreground_mask > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            top = max(0, min(y - box_h // 2, grid_h - box_h))
            left = max(0, min(x - box_w // 2, grid_w - box_w))
            box = torch.zeros_like(foreground_mask, dtype=torch.bool)
            box[top : top + box_h, left : left + box_w] = True
            if (box & foreground_mask.bool()).any():
                valid_coords.append((top, left))

        if not valid_coords:
            return torch.zeros_like(foreground_mask, dtype=torch.bool)

        top, left = random.choice(valid_coords)
        region_mask = torch.zeros_like(foreground_mask, dtype=torch.bool)
        region_mask[top : top + box_h, left : left + box_w] = True
        return region_mask & foreground_mask.bool()

    @staticmethod
    def mask_to_patch(mask: np.ndarray, grid_h: int, grid_w: int):
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(grid_h, grid_w), mode="nearest")
        return (mask_tensor.squeeze(0).squeeze(0) > 0.5)

    @staticmethod
    def get_spatial_neighbors(token_idx: int, grid_h: int, grid_w: int, radius: int):
        y, x = divmod(token_idx, grid_w)
        neigh = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    neigh.append(ny * grid_w + nx)
        return neigh

    def project_to_local_manifold(self, local_feats: torch.Tensor, token_feat: torch.Tensor):
        mu = local_feats.mean(dim=0, keepdim=True)
        centered = local_feats - mu
        q = min(self.pca_dim, centered.shape[0], centered.shape[1])

        if q < 1:
            return token_feat, torch.zeros_like(token_feat)

        _, _, basis = torch.pca_lowrank(centered, q=q, center=False)
        token_centered = token_feat - mu.squeeze(0)
        coeff = token_centered @ basis
        proj = coeff @ basis.transpose(0, 1) + mu.squeeze(0)
        delta = token_feat - proj
        return proj, delta

    def get_decisive_dims(self, delta: torch.Tensor):
        feat_dim = delta.numel()
        k = max(self.channel_min, int(round(feat_dim * self.channel_topk_ratio)))
        k = min(k, feat_dim)
        if k <= 0:
            return torch.empty(0, device=delta.device, dtype=torch.long)
        return torch.topk(delta.abs(), k=k, largest=True).indices
