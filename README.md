
Requirements

1.	Clone the Repository

```bash
git clone https://github.com/aswahd/SamRadiology.git
cd sam2rad
```


2.	Set Up a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3.	Install Dependencies

```bash
pip install -r requirements.txt
```
at https://github.com/Dao-AILab/causal-conv1d/releases install causal-conv1d
at https://github.com/state-spaces/mamba/releases install mamba-ssm

4.	Download Pre-trained Weights
Download the pre-trained weights from the official SAM repository and place them in the weights directory:


## Quickstart

File structure:
```markdown
root
├── Train
│   ├── imgs
            ├── 1.png
            ├── 2.png
            ├── ...
            |
│   └── gts
            ├── 1.png
            ├── 2.png
            ├── ...
└── Test
    ├── imgs
            ├── 1.png
            ├── 2.png
            ├── ...
    └── gts
            ├── 1.png
            ├── 2.png
            ├── ...
```


Download Sample Dataset:
- Download the preprocessed data from [ACDC dataset](https://drive.google.com/drive/folders/14WIOWTF1WWwMaHV7UVo5rjWujpUxGetJ?usp=sharing).
- Extract the data to `./datasets/ACDCPreprocessed`.



## Models

Sam2Rad supports various image encoders and mask decoders, allowing flexibility in model architecture.

**Supported Image Encoders**
-	sam_vit_b_adapter
-	sam_vit_l_adapter
-	sam_vit_h_adapter
-	sam_vit_b
-	sam_vit_l
-	sam_vit_h
-	vit_tiny
-	All versions of Sam2 image encoder with or without adapters

All supported image encoders are available in the [sam2rad/encoders/build_encoder.py](sam2rad/encoders/build_encoder.py).

**Supported Mask Decoders**

-	sam_mask_decoder
-	lora_mask_decoder
-	All versions of Sam2 mask decoder


All supported mask decoders are available in the [sam2rad/decoders/build_decoder.py](sam2rad/decoders/build_decoder.py).

## Training
Prepare a configuration file for training. Here is an example configuration file for training on the ACDC dataset:

```yaml
image_size: 1024
image_encoder: "sam_vit_b_adapter"
mask_decoder: "lora_mask_decoder"
sam_checkpoint: "weights/sam_vit_b.pth"
wandb_project_name: "MSD"
prompt_predictor: "sam_vitb_high_res_ppn"

dataset:
  name: MSD
  root: /dataNfs/ncp/SamRadiology-main/dataset/MSD
  image_size: 512
  split: 0.88
  seed: 42
  batch_size: 8
  num_workers: 4
  num_classes: 1
  num_tokens: 10

training:
  max_epochs: 400
  save_path: checkpoints/msd

inference:
  name: msd_test
  root: /dataNfs/ncp/SamRadiology-main/msd/test
  checkpoint_path: checkpoints/msd/XXX.pt
```


```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/your/config.yaml

```
Replace `/path/to/your/config.yaml` with the actual path to your configuration file.

## Evaluation

Ensure your configuration file points to the correct checkpoint and data paths:

```yaml
inference:
  
  image_size: 1024
  image_encoder: "sam_vit_b_adapter"
  mask_decoder: "lora_mask_decoder"
  sam_checkpoint: "weights/sam_vit_b.pth"
  wandb_project_name: "MSD"
  prompt_predictor: "sam_vitb_high_res_ppn"

  inference:
    model_checkpoint: "checkpoints/model_epoch=499-val_dice=0.95-v1.ckpt"
    input_images: /dataNfs/ncp/SamRadiology-main/dataset/MSD
    output_dir: /dataNfs/ncp/SamRadiology-main/logs/result/dataset/IMSD
    image_size: 512
    name: MSD_test

  dataset:
    name: isic2016
    root: /dataNfs/ncp/SamRadiology-main/dataset/ISIC2016
    image_size: 512
    split: 0.75
    seed: 42
    batch_size: 4
    num_workers: 4
    num_classes: 1
    num_tokens: 10
```
Run the evaluation script:
```bash
python -m sam2rad.evaluation.eval_bounding_box --config /path/to/your/config.yaml
python -m sam2rad.evaluation.eval_prompt_learner --config /path/to/your/config.yaml
```

