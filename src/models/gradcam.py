import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMAnomaly(nn.Module):
    def __init__(self, model, stats_path, device="cpu"):
        super().__init__()
        self.model = model.to(device).eval()
        self.device = device
        self.model.fc = nn.Identity()

        self.mean, self.cov_inv = self._load_stats(stats_path)
        self.activations = None
        self._register_hooks()

    def _load_stats(self, stats_path):
        stats = torch.load(stats_path, map_location=self.device, weights_only=False)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=self.device)
        cov_inv = torch.as_tensor(stats["cov_inv"], dtype=torch.float32, device=self.device)
        return mean, cov_inv

    def _register_hooks(self):
        self.model.layer2.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _mahalanobis_map(self, feature_map):
        h, w = feature_map.shape[-2:]
        flat = feature_map[0].permute(1, 2, 0).reshape(-1, feature_map.shape[1])
        diff = flat - self.mean.to(feature_map.device)
        scores = torch.einsum("pc, pcd, pd -> p", diff, self.cov_inv.to(feature_map.device), diff)
        return scores.reshape(h, w)

    def _normalize_cam(self, cam):
        cam = cam.detach().cpu().numpy()
        cam = np.asarray(cam, dtype=np.float32)
        cam = np.clip(cam, 0.0, None)
        cam = cam - cam.min()
        denom = cam.max() - cam.min()
        if denom < 1e-8:
            return np.zeros_like(cam)
        return cam / denom

    def forward(self, x):
        x = x.to(self.device)
        self.model.zero_grad()
        self.activations = None

        x = x.clone().detach().requires_grad_(True)
        self.model(x)

        if self.activations is None:
            raise RuntimeError("layer2 activations were not captured")

        captured_tensor = self.activations
        score_map = self._mahalanobis_map(captured_tensor)
        score = score_map.max()
        grad = torch.autograd.grad(score, captured_tensor, retain_graph=False, allow_unused=False)[0]

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * captured_tensor).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        return self._normalize_cam(cam[0, 0])

    def overlay_on_image(self, image_rgb, cam_map):
        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0.0, 1.0)
            image_rgb = (image_rgb * 255).astype(np.uint8)

        cam_map = np.asarray(cam_map, dtype=np.float32)
        cam_map = np.clip(cam_map, 0.0, 1.0)
        cam_map = cv2.resize(cam_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image_bgr, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR), 0.4, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
