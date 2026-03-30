## ProCap: Projection-Aware Captioning for Spatial Augmented Reality [IEEE VR'26]
  <p>
    <a href="https://github.com/ZimoCao/"><strong>Zimo Cao</strong></a><sup>1*</sup>
    &nbsp;&nbsp;
    <a href="https://Yu-chen-Deng.github.io/"><strong>Yuchen Deng</strong></a><sup>1*</sup>
    &nbsp;&nbsp;
    <a href="https://haibinling.github.io/"><strong>Haibin Ling</strong></a><sup>2</sup>
    &nbsp;&nbsp;
    <a href="https://bingyaohuang.github.io/"><strong>Bingyao Huang</strong></a><sup>1&dagger;</sup>
  </p>
  <p>
    <sup>1</sup>Southwest University,&nbsp;<sup>2</sup>Westlake University<br><sup>*</sup>Equal Contribution,&nbsp;<sup>&dagger;</sup>Corresponding Author
  </p>
  <br>
  <div align="center">
    <img src="./doc/merged.png" style="width:80%;">
  </div>

## Introduction

PyTorch's implementation of ProCap.

## Prerequisites
* PyTorch compatible GPU
* Conda

## Setup

1. Clone this repo:
   ```
   git clone https://github.com/zimocao/ProCap.git
   cd ProCap
   ```

2. Create a new conda environment:
   ```
   conda env create -f environment.yml
   conda activate ProCap
   ```

3. Download [RGBP dataset](#rgbp-dataset) and extract to a specific directory.

4. Modify `data_root` to RGBP dataset directory in `train_procap.py` and `eval_procap.py`.

## RGBP Dataset

RGBP (RGB + Projections), the first large-scale SAR semantic benchmark dataset, featuring 65 diverse physical scenes and over 180,000 projections with dense, decoupled annotations.

   ```
   RGBP_dataset_structure
    |
    |_ /train_coco/
    |_ /eval_coco/
    |_ /eval_nocaps/
    |_ /eval_whoops/
    |_ /plain/
    |_ /train_coco_scenes_1_to_60.jsonl
    |_ /eval_coco_scenes_1_to_60.jsonl
    |_ /eval_coco_scenes_61_to_65.jsonl
    |_ /eval_coco_scenes_66_to_70.jsonl
    |_ /eval_nocaps_scenes_1_to_60.jsonl
    |_ /eval_nocaps_scenes_61_to_65.jsonl
    |_ /eval_nocaps_scenes_66_to_70.jsonl
    |_ /eval_whoops_scenes_1_to_60.jsonl
    |_ /eval_whoops_scenes_61_to_65.jsonl
    |_ /eval_whoops_scenes_66_to_70.jsonl
   ```

> Note: the additional 5 unseen scenes mentioned in paper are indexed from 66 to 70. Further details can be found in our paper.

## Training

To train ProCap on the RGBP dataset, using the [`scripts/train_procap.sh`](scripts/train_procap.sh) script. For example:
   ```bash
   bash scripts/train_procap.sh \
     --gpus 1,2,3 \
     --model_type openlm-research/open_llama_3b
   ```

## Evaluation

To evaluate ProCap on the RGBP dataset, using the [`scripts/eval_procap.sh`](scripts/eval_procap.sh) script. For example:
   ```bash
   bash scripts/eval_procap.sh \
     --gpus 1,2,3 \
     --model_type openlm-research/open_llama_3b \
     --dataset coco/nocaps/whoops \
     --task scene/projection/all \
     --seen_scene/--unseen_scene
   ```

The required evaluation directory can be downloaded from the [Releases](https://github.com/FeiElysia/ViECap/releases/tag/checkpoints) of [ViECap](https://github.com/FeiElysia/ViECap) repository. (we thank the authors for their effort!)

## Reproduction
To reproduce the results reported in our paper, run:
   ```bash
   (ProCap) python reproduce_paper_results.py
   ```

## Citation

If you find our dataset or code helpful for your research, please kindly consider citing:

```bibtex
@ARTICLE{Cao2026ProCap,
  author    = {Cao, Zimo and Deng, Yuchen and Ling, Haibin and Huang, Bingyao},
  booktitle = {2026 IEEE Conference on Virtual Reality and 3D User Interfaces (in press)},
  title     = {ProCap: Projection-Aware Captioning for Spatial Augmented Reality},
  year      = {2026},
}
```

## Acknowledgments

- This code and external knowledge base borrow heavily from [EVCap](https://github.com/Jiaxuan-Li/EVCap), we thank the authors for their great effort.
- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
- We thank the authors of the colorful textured sampling images.
- Feel free to open an issue if you have any questions/suggestions/concerns 😁.

## License

The code related to the ProCap algorithm and RGBP dataset are available free of charge for non-commercial, non-profit use and may be redistributed under the terms specified in [license](LICENSE).

However, this project is largely built upon the open-source library [Diffusers](https://github.com/huggingface/diffusers), which is distributed under [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE). We gratefully acknowledge the community’s contributions.
