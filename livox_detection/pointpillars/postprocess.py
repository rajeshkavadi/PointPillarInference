"""CPU replacement for OpenPCDet's CUDA post-processing.

Implements, on CPU:

  * sigmoid + per-anchor max class score (class-agnostic NMS path),
  * score filtering, top-K pre-maxsize,
  * rotated 3D IoU: BEV polygon intersection area (Sutherland-Hodgman) times
    height overlap, divided by the 3D union volume -- identical to
    ``iou3d_nms_cuda.boxes_overlap_bev_gpu`` (BEV intersection AREA, not IoU)
    plus the height/union math in ``pcdet/ops/iou3d_nms/iou3d_nms_utils.py``,
  * greedy NMS with NMS_PRE/POST_MAXSIZE.

No CUDA, no compiled iou3d_nms extension.
"""

import numpy as np
import torch

_QUAD = np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
], dtype=np.float32)  # counter-clockwise


def boxes_to_corners_3d(boxes):
    """OpenPCDet corner order: (N, 8, 3) for boxes [x,y,z,dx,dy,dz,heading]."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ], dtype=np.float32) / 2.0
    corners = boxes[:, None, 3:6] * template[None, :, :]
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    x = corners[:, :, 0] * c[:, None] - corners[:, :, 1] * s[:, None]
    y = corners[:, :, 0] * s[:, None] + corners[:, :, 1] * c[:, None]
    corners[:, :, 0] = x
    corners[:, :, 1] = y
    corners += boxes[:, None, 0:3]
    return corners


def boxes_to_corners_bev(boxes):
    """(N, 4, 2) counter-clockwise BEV corners."""
    x, y = boxes[:, 0], boxes[:, 1]
    dx, dy = boxes[:, 3], boxes[:, 4]
    heading = boxes[:, 6]
    quad = torch.as_tensor(_QUAD, dtype=boxes.dtype, device=boxes.device)
    lx = quad[None, :, 0] * dx[:, None]
    ly = quad[None, :, 1] * dy[:, None]
    cos, sin = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
    rx = lx * cos - ly * sin + x[:, None]
    ry = lx * sin + ly * cos + y[:, None]
    return torch.stack([rx, ry], dim=-1)


def _polygon_area(poly, valid):
    counts = valid.sum(dim=1)
    idx = torch.arange(poly.shape[1], device=poly.device).unsqueeze(0)
    nxt = torch.where(idx + 1 < counts.unsqueeze(1), idx + 1, torch.zeros_like(idx))
    x, y = poly[..., 0], poly[..., 1]
    xn = torch.gather(x, 1, nxt)
    yn = torch.gather(y, 1, nxt)
    cross = x * yn - xn * y
    return 0.5 * torch.abs((cross * valid.to(cross.dtype)).sum(dim=1))


def _intersection_area(poly_a, poly_b):
    """Vectorised Sutherland-Hodgman intersection of P convex quadrilaterals."""
    p = poly_a.shape[0]
    device = poly_a.device
    poly = poly_a
    valid = torch.ones((p, 4), dtype=torch.bool, device=device)

    for edge_idx in range(4):
        s = poly.shape[1]
        a = poly_b[:, edge_idx, :].unsqueeze(1)
        b = poly_b[:, (edge_idx + 1) % 4, :].unsqueeze(1)
        edge = b - a
        rel = poly - a
        cross = edge[:, :, 0] * rel[:, :, 1] - edge[:, :, 1] * rel[:, :, 0]
        inside = (cross >= 0) & valid

        counts = valid.sum(dim=1)
        idx = torch.arange(s, device=device).unsqueeze(0)
        prev = torch.where(idx == 0, (counts - 1).unsqueeze(1), idx - 1)
        prev = prev.clamp(0, s - 1)
        poly_prev = torch.gather(poly, 1, prev.unsqueeze(-1).expand(-1, -1, 2))
        inside_prev = torch.gather(inside, 1, prev)
        valid_prev = torch.gather(valid, 1, prev)

        crossing = (inside != inside_prev) & valid & valid_prev
        seg = poly - poly_prev
        denom = seg[..., 0] * edge[:, :, 1] - seg[..., 1] * edge[:, :, 0]
        d1 = poly_prev - a
        numer = -(d1[..., 0] * edge[:, :, 1] - d1[..., 1] * edge[:, :, 0])
        safe = denom.abs() > 1e-9
        t = torch.where(safe, numer / torch.where(safe, denom, torch.ones_like(denom)),
                        torch.zeros_like(denom))
        inter = poly_prev + t.clamp(0.0, 1.0).unsqueeze(-1) * seg

        enter = crossing & inside
        leave = crossing & (~inside)
        emit_curr = inside
        slot0 = torch.where(enter.unsqueeze(-1), inter, poly)
        slot0 = torch.where(leave.unsqueeze(-1), inter, slot0)
        valid0 = (enter | emit_curr | leave) & valid
        slot1 = poly
        valid1 = enter & valid

        base0 = (2 * torch.arange(s, device=device)).unsqueeze(0).expand(p, -1)
        base1 = base0 + 1
        out = poly.new_zeros((p, 2 * s, 2))
        out_valid = torch.zeros((p, 2 * s), dtype=torch.bool, device=device)
        out.scatter_(1, base0.unsqueeze(-1).expand(-1, -1, 2), slot0)
        out.scatter_(1, base1.unsqueeze(-1).expand(-1, -1, 2), slot1)
        out_valid.scatter_(1, base0, valid0)
        out_valid.scatter_(1, base1, valid1)

        total = 2 * s
        key = torch.arange(total, device=device, dtype=torch.float32).unsqueeze(0)
        key = key + (~out_valid).to(torch.float32) * float(total)
        perm = torch.argsort(key, dim=1)
        poly = torch.gather(out, 1, perm.unsqueeze(-1).expand(-1, -1, 2))
        valid = torch.gather(out_valid, 1, perm)
        ar = torch.arange(total, device=device).unsqueeze(0)
        valid = valid & (ar < out_valid.sum(dim=1).unsqueeze(1))
    return _polygon_area(poly, valid)


def boxes_overlap_bev(boxes_a, boxes_b, grid_chunk=512, pair_chunk=8192):
    """BEV intersection AREA matrix [Na, Nb].

    This matches OpenPCDet's ``iou3d_nms_cuda.boxes_overlap_bev_gpu``: the
    returned value is intersection area, NOT BEV IoU.  A cheap circumscribed-
    circle test drops non-overlapping pairs before polygon clipping.
    """
    na, nb = boxes_a.shape[0], boxes_b.shape[0]
    overlap = boxes_a.new_zeros((na, nb))
    if na == 0 or nb == 0:
        return overlap

    ca = boxes_to_corners_bev(boxes_a)
    cb = boxes_to_corners_bev(boxes_b)
    radius_a = 0.5 * torch.sqrt(boxes_a[:, 3] ** 2 + boxes_a[:, 4] ** 2)
    radius_b = 0.5 * torch.sqrt(boxes_b[:, 3] ** 2 + boxes_b[:, 4] ** 2)

    for i0 in range(0, na, grid_chunk):
        i1 = min(i0 + grid_chunk, na)
        for j0 in range(0, nb, grid_chunk):
            j1 = min(j0 + grid_chunk, nb)
            dx = boxes_a[i0:i1, 0:1] - boxes_b[j0:j1, 0].unsqueeze(0)
            dy = boxes_a[i0:i1, 1:2] - boxes_b[j0:j1, 1].unsqueeze(0)
            reach = radius_a[i0:i1].unsqueeze(1) + radius_b[j0:j1].unsqueeze(0)
            cand = (dx * dx + dy * dy) <= (reach * reach)
            if not cand.any():
                continue
            ii, jj = cand.nonzero(as_tuple=True)
            for k0 in range(0, ii.shape[0], pair_chunk):
                k1 = min(k0 + pair_chunk, ii.shape[0])
                ia = ii[k0:k1] + i0
                ib = jj[k0:k1] + j0
                overlap[ia, ib] = _intersection_area(ca[ia], cb[ib])
    return overlap


def boxes_iou_bev(boxes_a, boxes_b, **kwargs):
    """Rotated BEV IoU matrix [Na, Nb] (intersection area / union area)."""
    overlap = boxes_overlap_bev(boxes_a, boxes_b, **kwargs)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).clamp(min=1e-6).view(-1, 1)
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).clamp(min=1e-6).view(1, -1)
    union = area_a + area_b - overlap
    return (overlap / union.clamp(min=1e-6)).clamp(0.0, 1.0)


def boxes_iou3d(boxes_a, boxes_b, **kwargs):
    """3D IoU: rotated BEV overlap * height overlap / 3D union volume."""
    overlaps_bev = boxes_overlap_bev(boxes_a, boxes_b, **kwargs)

    a_zmin = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    a_zmax = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    b_zmin = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    b_zmax = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    overlap_h = torch.clamp(torch.min(a_zmax, b_zmax) - torch.max(a_zmin, b_zmin), min=0)

    overlap_3d = overlaps_bev * overlap_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    return overlap_3d / torch.clamp(vol_a + vol_b - overlap_3d, min=1e-6)


def nms_3d(boxes, scores, thresh, pre_maxsize=4096, post_maxsize=500):
    """Greedy class-agnostic 3D NMS; returns (selected_indices, scores)."""
    if boxes.shape[0] == 0:
        return (torch.zeros(0, dtype=torch.long, device=boxes.device),
                boxes.new_zeros((0,)))
    k = min(int(pre_maxsize), boxes.shape[0])
    top_scores, top_idx = torch.topk(scores, k=k)
    top_boxes = boxes[top_idx]

    overlaps = boxes_iou3d(top_boxes, top_boxes) > float(thresh)
    suppressed = torch.zeros(k, dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(k):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= int(post_maxsize):
            break
        suppressed |= overlaps[i]
        suppressed[i] = True
    keep_t = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    return top_idx[keep_t], top_scores[keep_t]


def postprocess(cls_logits, box_preds, score_thresh=0.1, nms_thresh=0.01,
                nms_pre_maxsize=4096, nms_post_maxsize=500):
    """OpenPCDet class-agnostic post-processing.

    Args:
        cls_logits: [1, N, num_class] raw logits.
        box_preds:  [1, N, 7] decoded boxes.
    Returns:
        boxes [K, 7], scores [K], labels [K] (1-based class ids).
    """
    scores_all = torch.sigmoid(cls_logits[0])
    scores, labels = torch.max(scores_all, dim=-1)
    labels = labels + 1
    boxes = box_preds[0]

    mask = scores >= float(score_thresh)
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    if boxes.shape[0] == 0:
        return boxes, scores, labels

    selected, selected_scores = nms_3d(
        boxes, scores, nms_thresh,
        pre_maxsize=nms_pre_maxsize, post_maxsize=nms_post_maxsize)
    return boxes[selected], selected_scores, labels[selected]
