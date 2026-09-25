# CNN Learning Path: from building blocks to Camouflaged Object Detection

Step-by-step PyTorch + OpenCV lessons. Every file runs on its own and ends with a **TRY IT** list of changes to make.

## Setup (Windows PC with an NVIDIA GPU)

```bash
# 1. Install PyTorch with CUDA. Pick your CUDA version at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Other libraries
pip install -r requirements.txt

# 3. Check that everything works (GPU + dataset)
cd cnn_learning
python 00_check_setup.py
```

Set your dataset path once in `config.py` (`DATASET_ROOT`). Lessons save pictures to `cnn_learning/outputs/`.

## Roadmap

| # | File | Topic | Status |
|---|------|-------|--------|
| 00 | `00_check_setup.py` | GPU, libraries, dataset layout | done |
| 01 | `01_conv_blocks.py` | Conv = OpenCV filter, shapes, ConvBNReLU, DoubleConv, Residual, Dilated, Depthwise | done |
| 02 | `02_encoder.py` | Stack blocks into an encoder, multi-scale features | next |
| 03 | `03_decoder.py` | Upsampling + skip connections (concat vs add) | |
| 04 | `04_unet.py` | Full UNet from scratch | |
| 05 | `05_resnet_encoder.py` | Pretrained ResNet50 as the encoder | |
| 06 | `06_unetpp.py` | UNet++ nested skips | |
| 07 | `07_train_cod.py` | Dataset loader, loss (BCE + IoU), metrics, training on COD | |
| 08 | `08_attention.py` | SE / CBAM attention modules | |
| 09 | `09_sinet.py` | SINet: RF module, search + identification | |
| 10 | `10_my_cod_net.py` | Your own COD architecture | |

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
