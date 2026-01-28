## Salsa as a Nonverbal Embodied Language–The CoMPAS3D Dataset and Benchmarks

### Overview
Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner’s proficiency, using haptic signaling as a primary form of communication. While today’s AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond words—it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation.

We present **CoMPAS3D**, the largest and most diverse motion-capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes:
- **3 hours** of leader–follower salsa dances
- **18 dancers** spanning beginner, intermediate, and professional skill levels
- **2,800+ move segments** annotated with move types, combinations, errors, and stylistic elements

We draw analogies between partner dance communication and natural language, defining two benchmark tasks for synthetic 3D humans that parallel key problems in spoken dialogue: speaker/listener synthesis (leader/follower generation) and duet (conversation) generation.

Alongside the dataset and expert annotations, we release:
1. A multitask **SalsaAgent** model capable of:
   - **Leader→Follower** and **Follower→Leader** motions
   - **Caption→Motion** generation
2. **Baselines** and **evaluation scripts** to accelerate research in socially interactive embodied AI.

---

## Getting Started

### 1. Environment Setup
```bash
conda create -n motionagent python=3.10
conda activate motionagent
pip install -r requirements.txt

# Note: If you encounter missing dependencies, you may need to install them manually:
# pip install lmdb human_body_prior roma body_visualizer

# Note: body_visualizer requires a display for visualization. If running on a headless server,
# you may need to set up X11 forwarding or use a virtual display.

## Current Setup Status
✅ Environment created and activated  
✅ Requirements installed  
✅ Pretrained checkpoints downloaded  
✅ WavTokenizer dependencies installed  
✅ human_body_prior installed  
✅ roma installed  
✅ body_visualizer installed locally  
✅ Model loading successful (past import and checkpoint issues)
✅ LMDB training data moved to correct location
✅ Fixed Windows-specific paths and arguments for Ubuntu
✅ WavTokenizer checkpoint downloaded (1.76GB)
⚠️ CUDA out of memory error (model requires >10.75GB GPU memory)
⚠️ Display/OpenGL context issues on headless server (common issue)

## Download WavTokenizer Checkpoint
The demo requires a WavTokenizer checkpoint file that is not included in the repository:
- **Required file**: `utils/salsa_utils/libs/WavTokenizer/results/train/wavtokenizer_large_unify_600_24k.ckpt`
- **Download command**:
  ```bash
  mkdir -p utils/salsa_utils/libs/WavTokenizer/results/train
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='novateur/WavTokenizer-large-unify-40token', filename='wavtokenizer_large_unify_600_24k.ckpt', local_dir='utils/salsa_utils/libs/WavTokenizer/results/train')"
  ```
- **Source**: [Hugging Face - WavTokenizer-large-unify-40token](https://huggingface.co/novateur/WavTokenizer-large-unify-40token/tree/main)

## Troubleshooting Display Issues
If you encounter OpenGL/display errors on a headless server:
1. Use X11 forwarding: `ssh -X username@server`
2. Set up virtual display: `export DISPLAY=:99 && Xvfb :99 -screen 0 1024x768x24 &`
3. Install Mesa software rendering: `sudo apt-get install mesa-utils`

## GPU Memory Requirements
The SalsaAgent model requires significant GPU memory:
- **Minimum**: 12GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Current issue**: CUDA out of memory (model needs >10.75GB)

### Solutions for GPU Memory Issues:
1. **Use CPU instead**: Set `--device cpu` (slower but no memory limits)
2. **Reduce batch size**: Modify batch size in options
3. **Use smaller model**: Switch to a smaller LLM backbone
4. **Gradient checkpointing**: Enable memory-efficient training
```

### 2. Download Pretrained Checkpoints & Assets
We provide helper scripts to fetch all required weights and models:
```bash
bash prepare/download_ckpt.sh      # SalsaAgent checkpoints
bash prepare/download_vqvae.sh     # VQ-VAE model
bash prepare/download_glove.sh     # GloVe embeddings
bash prepare/download_extractor.sh # Evaluation extractor models
```

### 3. Prepare Body Models & Auxiliary Data
1. **SMPL-X**: Download from [SMPL-X Website](https://smpl-x.is.tue.mpg.de/) and place in:
   ```
   ./body_model/smplx
   ```
2. **SMPL-H AMASS**: Download the AMASS SMPL-H dataset and place in:
   ```
   ./utils/salsa_utils/lib/MotionScript/data/smplh_amass
   ```
3. **MotionScript**: We use MotionScript for data augmentation during pretraining [motionscript](https://arxiv.org/pdf/2312.12634). Ensure the `data` folder contains the SMPL-H files above.

### 4. Install WaveTokenizer
We use a WaveTokenizer for audio tokenization at 40 tokens/sec:
1. The WavTokenizer is already included in the repository under `utils/salsa_utils/libs/WavTokenizer/`
2. Install its dependencies:
   ```bash
   cd utils/salsa_utils/libs/WavTokenizer
   pip install -r requirements.txt
   ```
3. Add to Python path (e.g., in `~/.bashrc`):
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)/utils/salsa_utils/libs/WavTokenizer"
   ```

---

## Data Preparation

### 1. Download Raw Dataset
Download the CoMPAS3D raw data, which includes:
- **SMPL-X extreme body models**
- **`synced_animation/`** folder with paired motion and audio (pre-synchronized)

Place under:
```bash
--salsa_data_root /path/to/CoMPAS3D
``` 

### 2. Generate LMDB
Run the script directly to process the raw dataset and create LMDB files:

**Important: HumanML3D Example File Required**
To be consistent with common motion datasets such as HumanML3D, the preprocessing pipeline:
1. Rotates keypoints **-90 degrees around X-axis** to align with HumanML3D coordinate system
2. Uses an example file from HumanML3D to compute target skeleton offsets (`tgt_offsets`) during preprocessing for producing HumanML3D representation

Before processing, you need to place an example file from HumanML3D processed `new_joints` folder:
- **Source**: Get `000021.npy` from HumanML3D processed `new_joints` folder
- **Destination**: Place it at `body_model/HML3D_Example_joonts/000021.npy`
- **Purpose**: This file is used to compute target skeleton offsets (`tgt_offsets`) for proper motion processing. Without it, the processing may produce NaN values in HML3D vectors.

```bash
python utils/salsa_utils/salsa_utils.py \
    --salsa_data_root /local-scratch/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/Dataset/compas3d \
    --save_path ./dataset_processed_New
```

This will create the LMDB files in `./dataset_processed_New/lmdb_Salsa_pair/lmdb_train` and `lmdb_test`.

### 3. Create Cache
After generating the LMDB, create a cached version for faster data loading:
```bash
python demo.py --create_cache_only --split train
```
This will create a cache at `./dataset_processed_New/lmdb_Salsa_pair/lmdb_train_cache_MDM`.

### 4. Run Visualization App
To visualize the dataset samples interactively:
```bash
python visualization/visualization_app.py
```
The app will launch on `http://0.0.0.0:7861` and allows you to:
- Browse samples from the LMDB
- Visualize leader and follower skeletons
- View combined animations
- Generate VQVAE reconstructions
- Visualize audio tokenization
- Experiment with relative motions

You may also use our provided processed data.
### 5. Create Training Samples
`Salsa_dataloader.py` reads the LMDB and yields minibatches:
```bash
python -c "from Salsa_dataloader import SalsaDataset; ds = SalsaDataset(lmdb_path='./data/salsa.lmdb')"
```

---

## Training

We use the same two-stage strategy as [Motion-Agent](https://github.com/szqwu/Motion-Agent): **Stage 1** trains on broad/multi-task data to learn a good motion–text representation; **Stage 2** fine-tunes on a single task (e.g. leader→follower or caption→motion) from the stage-1 checkpoint. See [LLM_PIPELINE_COMPARISON.md](LLM_PIPELINE_COMPARISON.md) for alignment with the base code and hyperparameter notes.

### Training MotionLLM

**Script**: `train_motionllm_salsa.py`

**Stage 1 (multi-task / representation)**  
Use HumanML3D-style data and Motion-Agent–like hyperparameters (e.g. `--lr 1e-5`, `--epochs 500`, `--train-batch-size 4`):

| Setting | Command |
|--------|--------|
| **HumanML3D, no audio** | `python train_motionllm_salsa.py --task caption_to_motion --lr 1e-5 --epochs 500 --train-batch-size 4 --save-dir output_trained/stage1_humanml3d` |
| **HumanML3D, with audio** | `python train_motionllm_salsa.py --task caption_to_motion --include-audio --lr 1e-5 --epochs 500 --train-batch-size 4 --save-dir output_trained/stage1_humanml3d_audio` |
| **InterHuman, no audio** | `python train_motionllm_salsa.py --motion-repr-type interhuman --task caption_to_motion --lr 1e-5 --epochs 500 --train-batch-size 4 --save-dir output_trained/stage1_interhuman` |
| **InterHuman, with audio** | `python train_motionllm_salsa.py --motion-repr-type interhuman --include-audio --task caption_to_motion --lr 1e-5 --epochs 500 --train-batch-size 4 --save-dir output_trained/stage1_interhuman_audio` |

**Stage 2 (task-specific)**  
Resume from a stage-1 (or pretrained) checkpoint and fine-tune on one task:

```bash
python train_motionllm_salsa.py --task leader_to_follower --resume-ckpt output_trained/stage1_humanml3d/Xmotionllm_epoch500.pth --lr 1e-5 --epochs 50 --save-dir output_trained/leader_to_follower
```

Optional: `--use-wandb` to log to Weights & Biases.

---

## Demo & Inference

Before running the demo, ensure your pretrained models (e.g. checkpoint files, VQ-VAE, GloVe, evaluator, and our fine-tuned for different tasks) are placed in a folder (e.g. `./checkpoints`) and point to it via `--model_ckpt`.

Run the interactive demo to load a model and perform inference for one of the multitask SalsaAgent capabilities.

```bash
python demo.py   --model_ckpt ./checkpoints/finetune/leader_to_follower.pt   --task follower_to_leader
```

If you omit `--task`, the script will prompt you to choose:

Supported tasks:

- `baseline`
- `follower_to_leader`
- `leader_to_follower`
- `caption_to_motion`

Example without manual prompt:

```bash
python demo.py --model_ckpt ./checkpoints/finetune/caption_to_motion.pt                --task caption_to_motion
```

---

## Citation & Acknowledgements
Please cite our work as:
> **Salsa as a Nonverbal Embodied Language–The CoMPAS3D Dataset and Benchmarks**

We thank the authors of:
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)
- [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)
- [Motion-Agent](https://github.com/modelscope/motionagent)
- [text-to-motion](https://github.com/EricGuo5513/text-to-motion)


