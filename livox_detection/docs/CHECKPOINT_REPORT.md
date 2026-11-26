# Checkpoint report

## Input

- User checkpoint: `/home/rajesh/live_inference_pp/pointpillar_7728_ft.pth`
- Copied to this project: `checkpoints/pointpillars_checkpoint.pth`
- Loaded on CPU with `torch.load(..., map_location="cpu")`; no CUDA device is requested anywhere.

## Identity / provenance

The file was compared with the official OpenPCDet KITTI PointPillars checkpoint
`pointpillar_7728.pth` (Google Drive id `1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm`):

| Check | Result |
|---|---|
| SHA-256 (user file) | `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2` |
| SHA-256 (official file) | `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2` |
| SHA-256 (project copy) | `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2` |
| `model_state` tensors | 127 |
| Tensor-by-tensor comparison | 127/127 bit-identical |
| `epoch` / `it` | 80 / 9280 |

Conclusion: this is the stock OpenPCDet KITTI `pointpillar` checkpoint, not a
custom architecture. The original KITTI model parameters are therefore used
unchanged in `configs/pointpillars.yaml`.

## Strict load result

`tools/pointpillars_cpu_inference.py --validate-only` reports:

```
device                : cpu (CPU only; CUDA is never used)
checkpoint            : strict load OK (127 tensors)
checkpoint meta       : {'epoch': 80, 'it': 9280}
class names           : ['Car', 'Pedestrian', 'Cyclist']
parameters            : 4834888 (4.835 M)
architecture          : PillarVFE -> PointPillarScatter -> BaseBEVBackbone -> AnchorHeadSingle
  vfe linear          : (64, 10)
  backbone blocks     : [13, 19, 19]
  head conv outputs   : cls=18 box=42 dir=12
point cloud range     : [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
voxel size            : [0.16, 0.16, 4.0]
BEV grid              : [432, 496, 1]
anchors / location    : 6
```

No missing or unexpected keys: `load_state_dict(..., strict=True)` succeeds.
No weight values or model parameters were edited.

## Sensor mismatch (documented, not hidden)

The checkpoint's VFE expects a 4th feature (intensity). The Livox bags contain
`x, y, z, rgb` and **no intensity field**. The default `INTENSITY_MODE=zero`
fills that feature with `0.0` and prints a loud warning. It does not relabel
`rgb` as intensity and does not change the model or config. For faithful KITTI
accuracy, use a point cloud that really contains an intensity field and set
`INTENSITY_MODE=field`.
