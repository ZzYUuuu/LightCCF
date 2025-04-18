# LightCCF
This is the PyTorch implementation for the paper:
> Yu Zhang, Yiwen Zhang^*, Yi Zhang, Lei Sang, Yun Yang ["Unveiling Contrastive Learning‘ Capability of Neighborhood Aggregation for Collaborative Filtering"](https://arxiv.org/pdf/2504.10113) in **ACM SIGIR 2025**, **July 13–17, 2025**, Padova, Italy.
## Environment Setting
```python
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
```
## Examples
We used three large-scale datasets: Douban-book, Amazon-book, and Tmall. Most of the parameters in LightCCF are fixed. We only need to adjust the temperature coefficient `t`, and `ssl_lambda`.
## Run
The best parameters for each dataset are provided in the `LightCCF.sh` file. You can find the corresponding code to run for each dataset in this file.
## Citation
If you find this work helpful, please cite it:
```
@article{yu_SUAU_ESWA_2024,
title = {Towards similar alignment and unique uniformity in collaborative filtering},
journal = {Expert Systems with Applications},
volume = {259},
pages = {125346},
year = {2025},
issn = {0957-4174},
author = {Lei Sang and Yu Zhang and Yi Zhang and Honghao Li and Yiwen Zhang},
}
```
