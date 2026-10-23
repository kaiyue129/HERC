# Integration summary against uploaded SSDMoE-UNet

## Original files modified

1. `umamba/nnunetv2/nets/SSDMoEUNet.py`
   - decoder can optionally return its final two feature maps
   - SSDMoEUNet forwards the optional feature-return flag
   - standard behavior is unchanged

2. `umamba/nnunetv2/run/load_pretrained_weights.py`
   - when destination is HERC, same-task segmentation heads are loaded
   - new `herc.*` keys may be absent in a baseline checkpoint

## New files

- `umamba/nnunetv2/nets/HERCUNet.py`
- `umamba/nnunetv2/nets/herc/*`
- `umamba/nnunetv2/training/loss/herc_loss.py`
- `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerHERC.py`
- `evaluation/herc_metrics.py`
- `umamba/nnunetv2/tests/herc/test_herc_core.py`

## Files deliberately not changed

- dataloaders
- preprocessing
- data normalization
- nnU-Net region transform
- sliding-window inference
- Gaussian aggregation
- export code
- fold splitting
- SSD-MoE implementation
- SoftMoE implementation
- GSAF implementation
