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
1. Clone & install per instructions:
2. Follow the instruction provided here: [https://github.com/jishengpeng/WavTokenizer/tree/main](https://github.com/jishengpeng/WavTokenizer/tree/main)
   ```bash
   git clone https://github.com/YourOrg/WaveTokenizer.git utils/salsa_utils/lib/wavetokenizer
   cd utils/salsa_utils/lib/wavetokenizer
   python setup.py install
   ```
3. Add to Python path (e.g., in `~/.bashrc`):
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)/utils/salsa_utils/lib/wavetokenizer"
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
Use `SFU_salsa_dance.py` to build a fast LMDB for training:
```bash
python SFU_salsa_dance.py   --salsa_data_root /path/to/CoMPAS3D   --output_lmdb ./data/salsa.lmdb
```
You may also use our provided processed data.
### 3. Create Training Samples
`Salsa_dataloader.py` reads the LMDB and yields minibatches:
```bash
python -c "from Salsa_dataloader import SalsaDataset; ds = SalsaDataset(lmdb_path='./data/salsa.lmdb')"
```

---

## Training

### Pretraining (None Task)
Run the general MotionLLM pretraining with:
```bash
python Train_motionllm_sals.py   --task none   --data_root ./data/salsa.lmdb   --save_dir ./checkpoints/pretrain
```

### Fine-tuning
Resume from the pretraining checkpoint:
```bash
python Train_motionllm_sals.py   --task finetune   --pretrained_ckpt ./checkpoints/pretrain/latest.pt   --data_root ./data/salsa.lmdb   --save_dir ./checkpoints/finetune
```

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

MotionScript: Z. Wu et al., *MotionScript: Visual Specification of Human Motion Synthesis*, ACM SIGGRAPH Asia 2023.
