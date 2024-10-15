# LightCCF
## Environment Setting
```python
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
```
## Examples
We used three large-scale datasets: Douban-book, Amazon-book, and Tmall. Most of the parameters in LightCCF are fixed. We only need to adjust the temperature coefficient `t`, `ssl_lambda`.
## Run
The best parameters for each dataset are provided in the `.sh` file.
