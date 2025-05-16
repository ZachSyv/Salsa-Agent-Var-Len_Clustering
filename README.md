# Salsa as a Nonverbal Embodied Language–The CoMPAS3D Dataset and Benchmarks 

## Overview

Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text—it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. Salsa’s formal judging standards offer evaluation criteria that are uncommon in other expressive interactions, making it particularly suitable for benchmarking embodied social AI. We draw analogies between partner dance communication and natural language, defining two benchmark tasks for synthetic 3D humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation.



## Getting Started

### Environment Setup
```bash
conda create -n motionagent python=3.10
conda activate motionagent
pip install -r requirements.txt
```
### Download Salsa-Agent ckpts
Download Salsa-Agent ckpts.
```bash
bash prepare/download_ckpt.sh
```
### Download Glove and extractor
Download evaluation models and gloves for evaluation.
```bash
bash prepare/download_glove.sh
bash prepare/download_extractor.sh
```

### Prepare the LLM backbone
We use Google Gemma2-2B as MotionLLM's backbone. Please grant access from [huggingface](https://huggingface.co/google/gemma-2-2b) and use `huggingface-cli login` to login.

## Demo
We provide demo for Salsa-Agent that uses our test set. You will need to download the data and preprocess them or you could easily download preprocessed data.
To start the demo:

```bash
python demo.py
```

### Example Prompts
Here are some examples of what you can do with Salsa-Agent:

1. **Solo Dance generation**
```bash
python demo.py -- caption_to_motion
```

2. **Duet Dance Generation**
```bash
python demo.py -- leader_to_follower
python demo.py -- follower_to_leader
```
<details>
<summary>Preview of the example motion</summary>


## Evaluation
Please refer to ```metrics``` folder for the evaluation details.


## Acknowledgements
We would like to thank the following open-source projects for their contributions to our codes:
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT),
[NExT-GPT](https://github.com/NExT-GPT/NExT-GPT),
[Motion-Agent](https://github.com/modelscope/motionagent),
[text-to-motion](https://github.com/EricGuo5513/text-to-motion).



