# CNN Learning Path: from building blocks to Camouflaged Object Detection

Step-by-step PyTorch + OpenCV lessons as Jupyter notebooks. Run each one top to bottom (**Kernel → Restart & Run All**), then do the **✏️ TRY IT** boxes: change the code, re-run the cell, and see what changes.

## Setup (Windows PC with an NVIDIA GPU)

```bash
# 1. Install PyTorch with CUDA. Pick your CUDA version at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Other libraries
pip install -r requirements.txt

# 3. Open the notebooks (or open the folder in VS Code with the Jupyter extension)
cd cnn_learning
jupyter notebook
```

Start with `00_check_setup.ipynb` to check your GPU and dataset.

Set your dataset path once in `config.py` (`DATASET_ROOT`). The notebooks import it, so keep `config.py` in the same folder as the notebooks.

Shared helper files (keep them next to the notebooks):
- `config.py`: dataset path, image size, device, `cod10k_pairs()` (image/mask pairs, camouflaged images only)
- `blocks.py`: the Lesson 01 bricks (`ConvBNReLU`, `DoubleConv`, `ResidualBlock`, `DepthwiseSeparableConv`) and `show()`

## Roadmap

| # | File | Topic | Status |
|---|------|-------|--------|
| 00 | `00_check_setup.ipynb` | GPU, libraries, dataset layout | done |
| 01 | `01_conv_blocks.ipynb` | Conv = OpenCV filter, shapes, ConvBNReLU, DoubleConv, Residual, Dilated, Depthwise | done |
| 02 | `02_encoder.ipynb` | UNet encoder, configurable encoder (the knobs), pretrained ResNet50, GPU budget | done |
| 03 | `03_decoder.ipynb` | Upsampling + skip connections (concat vs add) | next |
| 04 | `04_unet.ipynb` | Full UNet from scratch | |
| 05 | `05_resnet_unet.ipynb` | UNet with a pretrained ResNet50 encoder, freezing / fine-tuning | |
| 06 | `06_unetpp.ipynb` | UNet++ nested skips | |
| 07 | `07_train_cod.ipynb` | Dataset loader, loss (BCE + IoU), metrics, training on COD | |
| 08 | `08_attention.ipynb` | SE / CBAM attention modules | |
| 09 | `09_sinet.ipynb` | SINet: RF module, search + identification | |
| 10 | `10_my_cod_net.ipynb` | Your own COD architecture | |

## Mental model

```
image -> [ENCODER] -> [BOTTLENECK] -> [DECODER] -> mask
            |                            ^
            +------ skip connections ----+
```

- **Encoder**: H,W shrink and channels grow. It learns *what* is in the image.
- **Decoder**: H,W grow back. It rebuilds *where* the object is.
- **Skips**: pass fine detail from encoder to decoder.

Most new architectures, SINet included, change one of these four parts.
