"""Standalone OpenPCDet PointPillars model for CPU inference.

Parameter names are kept byte-for-byte identical to OpenPCDet:

    global_step, vfe.*, backbone_2d.*, dense_head.*

Only standard torch/numpy operations are used: no spconv, no CUDA extensions,
no .cuda() calls.  The forward path is a dense-CPU equivalent of OpenPCDet's
sparse 2D convolution network (for a 2D BEV grid, a sparse conv inference pass
is mathematically identical to the corresponding dense Conv2d pass).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pillar feature encoder: pcdet/models/backbones_3d/vfe/pillar_vfe.py
# ---------------------------------------------------------------------------
class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super(PFNLayer, self).__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            pieces = [self.linear(inputs[n * self.part:(n + 1) * self.part])
                      for n in range(num_parts + 1)]
            x = torch.cat(pieces, dim=0)
        else:
            x = self.linear(inputs)

        # Upstream toggles cudnn for BatchNorm1d; irrelevant on CPU.
        if self.use_norm:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)


class PillarVFE(nn.Module):
    def __init__(self, num_point_features, voxel_size, point_cloud_range,
                 use_norm=True, with_distance=False, use_absolute_xyz=True,
                 num_filters=(64,)):
        super(PillarVFE, self).__init__()
        self.use_norm = use_norm
        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz

        nfeat = num_point_features + (6 if use_absolute_xyz else 3)
        if with_distance:
            nfeat += 1
        num_filters_all = [nfeat] + list(num_filters)

        pfn_layers = []
        for i in range(len(num_filters_all) - 1):
            pfn_layers.append(PFNLayer(
                num_filters_all[i], num_filters_all[i + 1], use_norm,
                last_layer=(i >= len(num_filters_all) - 2),
            ))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = float(voxel_size[0])
        self.voxel_y = float(voxel_size[1])
        self.voxel_z = float(voxel_size[2])
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    @staticmethod
    def _paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        shape = [1] * len(actual_num.shape)
        shape[axis + 1] = -1
        max_num_t = torch.arange(max_num, dtype=torch.int,
                                 device=actual_num.device).view(shape)
        return actual_num.int() > max_num_t

    def forward(self, voxel_features, voxel_num_points, coords):
        # voxel_features: [M, T, 4], coords: [M, 4] = (batch, z, y, x)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True))
        features = torch.cat(features, dim=-1)

        mask = self._paddings_indicator(voxel_num_points, features.shape[1], axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features = features * mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.reshape(-1, features.shape[-1])


# ---------------------------------------------------------------------------
# Pillar scatter: pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py
# ---------------------------------------------------------------------------
class PointPillarScatter(nn.Module):
    def __init__(self, num_bev_features, grid_size):
        super(PointPillarScatter, self).__init__()
        self.num_bev_features = int(num_bev_features)
        self.nx, self.ny, self.nz = [int(v) for v in grid_size]
        assert self.nz == 1

    def forward(self, pillar_features, coords):
        if coords.shape[0] == 0:
            return pillar_features.new_zeros(
                (1, self.num_bev_features * self.nz, self.ny, self.nx))
        batch_size = int(coords[:, 0].max().item()) + 1
        maps = []
        for batch_idx in range(batch_size):
            spatial = torch.zeros(
                self.num_bev_features, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype, device=pillar_features.device)
            mask = coords[:, 0] == batch_idx
            this_coords = coords[mask]
            indices = (this_coords[:, 1] + this_coords[:, 2] * self.nx +
                       this_coords[:, 3]).long()
            spatial[:, indices] = pillar_features[mask].t()
            maps.append(spatial)
        out = torch.stack(maps, 0)
        return out.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)


# ---------------------------------------------------------------------------
# BEV backbone: pcdet/models/backbones_2d/base_bev_backbone.py
# ---------------------------------------------------------------------------
class BaseBEVBackbone(nn.Module):
    def __init__(self, num_bev_features, layer_nums, layer_strides, num_filters,
                 upsample_strides, num_upsample_filters,
                 use_conv_for_no_stride=False):
        super(BaseBEVBackbone, self).__init__()
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filters)

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        c_in_list = [int(num_bev_features)] + [int(v) for v in num_filters[:-1]]

        for idx in range(len(layer_nums)):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3,
                          stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[idx]):
                layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3,
                              padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*layers))

            stride = int(upsample_strides[idx])
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx],
                                       stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))
            else:
                stride = int(np.round(1.0 / stride))
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(num_filters[idx], num_upsample_filters[idx],
                              stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

        self.num_bev_features = int(sum(num_upsample_filters))

    def forward(self, spatial_features):
        x = spatial_features
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        x = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return x


# ---------------------------------------------------------------------------
# Anchor generation: pcdet/models/dense_heads/target_assigner/anchor_generator.py
# ---------------------------------------------------------------------------
def generate_anchors(anchor_range, anchor_generator_config, feature_map_sizes):
    """Exact CPU copy of OpenPCDet AnchorGenerator.generate_anchors().

    Returns (list of tensors shaped [fz, fy, fx, nsize, nrot, 7],
             list of anchors-per-location counts).
    """
    all_anchors = []
    counts = []
    for grid_size, cfg in zip(feature_map_sizes, anchor_generator_config):
        sizes = cfg["anchor_sizes"]
        rotations = cfg["anchor_rotations"]
        heights = cfg["anchor_bottom_heights"]
        align_center = bool(cfg.get("align_center", False))
        counts.append(len(rotations) * len(sizes) * len(heights))

        if align_center:
            x_stride = (anchor_range[3] - anchor_range[0]) / grid_size[0]
            y_stride = (anchor_range[4] - anchor_range[1]) / grid_size[1]
            x_offset, y_offset = x_stride / 2, y_stride / 2
        else:
            x_stride = (anchor_range[3] - anchor_range[0]) / (grid_size[0] - 1)
            y_stride = (anchor_range[4] - anchor_range[1]) / (grid_size[1] - 1)
            x_offset, y_offset = 0, 0

        # Same arange call as OpenPCDet (the +1e-5 guard makes the last shift
        # included); only `.cuda()` is deliberately omitted.
        x_shifts = torch.arange(anchor_range[0] + x_offset, anchor_range[3] + 1e-5,
                                step=x_stride, dtype=torch.float32)
        y_shifts = torch.arange(anchor_range[1] + y_offset, anchor_range[4] + 1e-5,
                                step=y_stride, dtype=torch.float32)
        z_shifts = x_shifts.new_tensor(heights)
        rot_t = x_shifts.new_tensor(rotations)
        size_t = x_shifts.new_tensor(sizes)

        try:
            gx, gy, gz = torch.meshgrid(x_shifts, y_shifts, z_shifts, indexing="ij")
        except TypeError:  # torch < 1.10
            gx, gy, gz = torch.meshgrid([x_shifts, y_shifts, z_shifts])

        anchors = torch.stack((gx, gy, gz), dim=-1)  # [x, y, z, 3]
        anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, size_t.shape[0], 1)
        size_rep = size_t.view(1, 1, 1, -1, 3).repeat(
            *anchors.shape[0:3], 1, 1)
        anchors = torch.cat((anchors, size_rep), dim=-1)
        anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, rot_t.shape[0], 1)
        rot_rep = rot_t.view(1, 1, 1, 1, -1, 1).repeat(
            *anchors.shape[0:3], size_t.shape[0], 1, 1)
        anchors = torch.cat((anchors, rot_rep), dim=-1)  # [x,y,z,nsize,nrot,7]
        anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
        anchors[..., 2] += anchors[..., 5] / 2  # bottom -> center
        all_anchors.append(anchors)
    return all_anchors, counts


# ---------------------------------------------------------------------------
# Detection head: pcdet/models/dense_heads/anchor_head_single.py
# ---------------------------------------------------------------------------
class AnchorHeadSingle(nn.Module):
    def __init__(self, input_channels, num_class, grid_size, point_cloud_range,
                 anchor_generator_config, use_direction_classifier=True,
                 dir_offset=0.78539, dir_limit_offset=0.0, num_dir_bins=2):
        super(AnchorHeadSingle, self).__init__()
        self.num_class = int(num_class)
        self.dir_offset = float(dir_offset)
        self.dir_limit_offset = float(dir_limit_offset)
        self.num_dir_bins = int(num_dir_bins)

        self.anchor_cfg = anchor_generator_config
        self.anchors, counts = generate_anchors(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_config,
            feature_map_sizes=[
                [grid_size[0] // c["feature_map_stride"],
                 grid_size[1] // c["feature_map_stride"]]
                for c in anchor_generator_config
            ],
        )
        self.num_anchors_per_location = int(sum(counts))
        # OpenPCDet: torch.cat(self.anchors, dim=-3).  Not a registered buffer:
        # registering it would add a key to state_dict and break strict loading.
        self.all_anchors = torch.cat(self.anchors, dim=-3)
        self._anchors_device = None

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1)
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * 7, kernel_size=1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.num_dir_bins, kernel_size=1)
        else:
            self.conv_dir_cls = None

    def _anchors_on(self, device):
        if self._anchors_device != device:
            self.all_anchors = self.all_anchors.to(device)
            self._anchors_device = device
        return self.all_anchors

    @staticmethod
    def _limit_period(val, offset, period):
        return val - torch.floor(val / period + offset) * period

    @staticmethod
    def residual_decode(box_encodings, anchors):
        """pcdet/utils/box_coder_utils.py::ResidualCoder.decode_torch."""
        xa, ya, za = anchors[..., 0:1], anchors[..., 1:2], anchors[..., 2:3]
        dxa, dya, dza = anchors[..., 3:4], anchors[..., 4:5], anchors[..., 5:6]
        ra = anchors[..., 6:7]
        xt, yt, zt = (box_encodings[..., 0:1], box_encodings[..., 1:2],
                      box_encodings[..., 2:3])
        dxt, dyt, dzt = (box_encodings[..., 3:4], box_encodings[..., 4:5],
                         box_encodings[..., 5:6])
        rt = box_encodings[..., 6:7]
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        return torch.cat([
            xt * diagonal + xa,
            yt * diagonal + ya,
            zt * dza + za,
            torch.exp(dxt) * dxa,
            torch.exp(dyt) * dya,
            torch.exp(dzt) * dza,
            rt + ra,
        ], dim=-1)

    def forward_raw(self, spatial_features_2d):
        """Return raw (pre-decode) head outputs, shaped [B, N, ...]."""
        cls = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        direction = None
        if self.conv_dir_cls is not None:
            direction = self.conv_dir_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()

        batch_size = cls.shape[0]
        anchors = self._anchors_on(cls.device).reshape(-1, 7)
        num_anchors = anchors.shape[0]
        batch_anchors = anchors.view(1, -1, 7).repeat(batch_size, 1, 1)

        cls = cls.view(batch_size, num_anchors, -1).float()
        box = box.view(batch_size, num_anchors, -1)
        if direction is not None:
            direction = direction.view(batch_size, num_anchors, -1)
        return cls, box, direction, batch_anchors

    def decode(self, cls, box, direction, batch_anchors):
        """ResidualCoder + direction classifier (identical to OpenPCDet)."""
        boxes = self.residual_decode(box, batch_anchors)
        if direction is not None:
            labels = torch.max(direction, dim=-1)[1]
            period = 2 * np.pi / self.num_dir_bins
            dir_rot = self._limit_period(
                boxes[..., 6] - self.dir_offset, self.dir_limit_offset, period)
            boxes[..., 6] = (dir_rot + self.dir_offset +
                             period * labels.to(boxes.dtype))
        return cls, boxes

    def forward(self, spatial_features_2d):
        cls, box, direction, anchors = self.forward_raw(spatial_features_2d)
        return self.decode(cls, box, direction, anchors)


# ---------------------------------------------------------------------------
# Full detector
# ---------------------------------------------------------------------------
class PointPillar(nn.Module):
    """Inference-only OpenPCDet PointPillar with identical parameter names."""

    def __init__(self, cfg):
        super(PointPillar, self).__init__()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self.grid_size = tuple(cfg["grid_size"])
        pcr = list(cfg["point_cloud_range"])
        voxel_size = list(cfg["voxel_size"])
        vfe_cfg = cfg["vfe"]

        self.vfe = PillarVFE(
            num_point_features=cfg["num_point_features"],
            voxel_size=voxel_size,
            point_cloud_range=pcr,
            use_norm=vfe_cfg["use_norm"],
            with_distance=vfe_cfg["with_distance"],
            use_absolute_xyz=vfe_cfg["use_absolute_xyz"],
            num_filters=vfe_cfg["num_filters"],
        )
        self.map_to_bev = PointPillarScatter(cfg["num_bev_features"], self.grid_size)

        b = cfg["backbone"]
        self.backbone_2d = BaseBEVBackbone(
            num_bev_features=cfg["num_bev_features"],
            layer_nums=b["layer_nums"],
            layer_strides=b["layer_strides"],
            num_filters=b["num_filters"],
            upsample_strides=b["upsample_strides"],
            num_upsample_filters=b["num_upsample_filters"],
        )

        h = cfg["head"]
        self.dense_head = AnchorHeadSingle(
            input_channels=self.backbone_2d.num_bev_features,
            num_class=cfg["num_class"],
            grid_size=self.grid_size,
            point_cloud_range=pcr,
            anchor_generator_config=h["anchor_generator_config"],
            use_direction_classifier=h["use_direction_classifier"],
            dir_offset=h["dir_offset"],
            dir_limit_offset=h["dir_limit_offset"],
            num_dir_bins=h["num_dir_bins"],
        )

    def extract_features(self, voxels, num_points, coords):
        """VFE -> scatter -> BEV backbone -> head convolutions."""
        features = self.vfe(voxels, num_points, coords)
        spatial = self.map_to_bev(features, coords)
        spatial_2d = self.backbone_2d(spatial)
        return self.dense_head.forward_raw(spatial_2d)

    def forward(self, voxels, num_points, coords):
        cls, box, direction, anchors = self.extract_features(voxels, num_points, coords)
        return self.dense_head.decode(cls, box, direction, anchors)
