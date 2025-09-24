
<h1 align=center>
  OOTSM: A Decoupled Linguistic Framework for Effective Scene Graph Anticipation
</h1>

<h3 align=center>
  ABSTRACT
</h3>

Abstract A scene graph is a structured represention of objects and their relationships in a scene. Scene Graph Anticipation (SGA) involves predicting future scene graphs from video clips, enabling applications as intelligent surveillance and human-machine collaboration. Existing SGA approaches primarily leverage visual cues, often struggling to integrate valuable commonsense knowledge, thereby limiting long-term prediction robustness. To explicitly leverage such commonsense knowledge, we propose a new approach to better understand the objects, concepts, and relationships in a scene graph. Our approach decouples the SGA task in two steps: first a scene graph capturing model is used to convert a video clip into a sequence of scene graphs, then a pure text-based model is used to predict scene graphs in future frames. Our focus in this work is on the second step, and we call it Linguistic Scene Graph Anticipation (LSGA) and believes it should have independent interest beyond the use in SGA discussed here. For LSGA, we introduce an Object-Oriented Two-Staged Method (OOTSM) where an Large Language Model (LLM) first forecasts object appearances and disappearances before generating detailed human-object relations. We conduct extensive experiments to evaluate OOTSM in two settings. For LSGA, we evaluate our fine-tuned open-sourced LLMs against zero-shot APIs (i.e., GPT-4o, GPT-4o-mini, and DeepSeek-V3) on a benchmark constructed from Action Genome annotations. For SGA, we combine our OOTSM with STTran++ from, and our experiments demonstrate effective state-of-the-art performance: short-term mean-Recall (@10) increases by 3.4% while long-term mean-Recall (@50) improves dramatically by 21.9%. Code is available in the supplementary material.

-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 

1. [SceneSayer](https://github.com/rohithpeddi/SceneSayer)
2. [STTran](https://github.com/yrcong/STTran)
3. [DSG-DETR](https://github.com/Shengyu-Feng/DSG-DETR)
4. [Tempura](https://github.com/sayaknag/unbiasedSGG)
5. [TorchDiffEq](https://github.com/rtqichen/torchdiffeq)
6. [TorchDyn](https://github.com/DiffEqML/torchdyn)


-------
# SETUP

## Dataset Preparation 

**Estimated time: 10 hours**

Follow the instructions from [here](https://github.com/JingweiJ/ActionGenome)

Download Charades videos ```data/ag/videos```

Download all action genome annotations ```data/ag/annotations```

Dump all frames ```data/ag/frames```

#### Change the corresponding data file paths in ```datasets/action_genome/tools/dump_frames.py```


Download object_bbox_and_relationship_filtersmall.pkl from [here](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view)
and place it in the data loader folder

### Install required libraries

```
conda create -n sga python=3.10 pip
conda activate sga
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

# Setup

### Build draw_rectangles modules

```
cd lib/draw_rectangles
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop draw_rectangles/
```

### Build bbox modules

```
cd fpn/box_intersections_cpu
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop fpn/box_intersections_cpu/
```

# fasterRCNN model

Remove any previous builds

``` 
cd fasterRCNN/lib
rm -rf build/
```

Change the folder paths in 'fasterRCNN/lib/faster_rcnn.egg.info/SOURCES.txt' to the current directory

```
python setup.py build develop
```

If there are any errors, check gcc version ``` Works for 9.x.x```


Follow [this](https://www.youtube.com/watch?v=aai42Qp6L28) for changing gcc version


Download pretrained fasterRCNN model [here](https://utdallas.box.com/s/gj7n57na15cel6y682pdfn7bmnbbwq8g) and place in fasterRCNN/models/

------

# Training and Testing

## Scene Graph Generation (SGG) Training

For training the scene graph generation model, please refer to our baseline implementation at [SceneSayer](https://github.com/rohithpeddi/SceneSayer). This includes the complete setup and training procedures for the SGG component.

## Training LLM Modules

Our approach uses two LLM modules that need to be trained separately:

1. First LLM Module (Object Prediction):
```bash
cd llama_SGA
bash run_SGA_0.sh
```

2. Second LLM Module (Relation Prediction):
```bash
cd llama_SGA
bash run_SGA_1.sh
```

## Testing

To run the complete evaluation pipeline:

```bash
bash run_test.sh
```

Pretrained checkpoints and weights will be released upon acceptance of the paper, in compliance with anonymity requirements.
