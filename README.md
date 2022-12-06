# Accurate BG
Repository for paper [Deep transfer learning and data augmentation improve glucose levels prediction in type 2 diabetes patients](https://www.nature.com/articles/s41746-021-00480-x).
![Setup](figs/setup.pdf)
## Installation
We recommend you create a virtual environment via `Anaconda` or `Pyenv`, then
activate the virtual environment and run
```
>> make dev
```
We used tensorflow 1.15.0.
## Train and test for the *OhioT1DM* dataset
To test for the OhioT1DM dataset, create a folder named `data` at the root directory
```
>> mkdir data/
```
Then, download the public dataset *OhioT1DM* [here](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html), and move the unzipped folder `OhioT1DM` into `data`.

To train and test for the *OhioT1DM* dataset, with our optimal configuration, run
```
>> cd accurate_bg
>> python3 ohio_main.py --epoch 150
```
the default prediction horizon is 6, equivalently 30 min. To adjust prediction horizon to 1hr, modify
the last line of command in the code block above with
```
>> python3 ohio_main.py --epoch 150 --prediction_horizon 12
```
## Reference
If you find this repo or our work helpful, we encourage you to cite the paper below.
```
@article{deng2021deep,
  title={Deep transfer learning and data augmentation improve glucose levels prediction in type 2 diabetes patients},
  author={Deng, Yixiang and Lu, Lu and Aponte, Laura and Angelidi, Angeliki M and Novak, Vera and Karniadakis, George Em and Mantzoros, Christos S},
  journal={NPJ Digital Medicine},
  volume={4},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```
