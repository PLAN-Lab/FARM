# FARM: Fine-Grained Alignment for Cross-Modal Recipe Retrieval

More information coming soon. Please feel free to reach out to us if you have any questions.

## Installation

First create the conda environment from the `env.yaml` file:

```
conda env create --name farm --file=env/env.yaml
source activate farm
```

Following T-Food, we use [bootstrap.pytorch](https://github.com/Cadene/bootstrap.pytorch.git).
```
cd bootstrap.pytorch
pip install -e .
```

Install [CLIP](https://github.com/openai/CLIP).
```
pip install git+https://github.com/openai/CLIP.git
```

## Dataset 
We use the [Recipe1M](http://im2recipe.csail.mit.edu/) dataset in this work.

## Evaluation
We adopt the evaluation code from [T-Food](https://github.com/mshukor/TFood). For the experiments with missing data, we use empty strings for the corresponding recipe components. Pre-trained models available in this [link](https://drive.google.com/drive/u/1/folders/1NuVdn_2RH9au0Z2NSoIhhWG_O4Q1s5vk).

## Acknowledgements
We would like to express our gratitude to [T-Food](https://github.com/mshukor/TFood) for their incredible work on the original project. We use their code for training and evaluating the models. The code for the hyperbolic embedding loss has been adopted from [UnHyperML](https://github.com/JiexiYan/UnHyperML).

## Citation
If you find this method and/or code useful, please consider citing:
```
@inproceedings{wahed2024fine,
  title={Fine-Grained Alignment for Cross-Modal Recipe Retrieval},
  author={Wahed, Muntasir and Zhou, Xiaona and Yu, Tianjiao and Lourentzou, Ismini},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5584--5593},
  year={2024}
}
```