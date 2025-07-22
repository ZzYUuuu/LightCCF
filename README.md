# LightCCF
This is the PyTorch implementation for the paper:
> Yu Zhang, Yiwen Zhang, Yi Zhang, Lei Sang, Yun Yang ["Unveiling Contrastive Learning‘ Capability of Neighborhood Aggregation for Collaborative Filtering"](https://arxiv.org/pdf/2504.10113). In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR’25).
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
@inproceedings{Yu_Unveiling_2025,
title = {Unveiling Contrastive Learning‘ Capability of Neighborhood Aggregation for Collaborative Filtering},
author={Zhang, Yu and Zhang, Yiwen and Zhang, Yi and Sang, Lei and Yang, Yun},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
doi={10.1145/3726302.3730111},
pages = {1985–1994},
numpages = {10},
year={2025},
}
```
