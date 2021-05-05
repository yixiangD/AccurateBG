# Accurate BG
Accurate blood glucose prediction for diabetic patients using deep learning
## Installation
We recommend you create a virtual environment via `Anaconda` or `Pyenv`, then
activate the virtual environment and run
```
>> make dev
```
We use tensorflow 1.15.0.
## Train and test for the *OhioT1DM* dataset
To test for the OhioT1DM dataset, create a folder named `data` at the root directory
```
>> mkdir data/
```
Then, download the public dataset *OhioT1DM* [here](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html), and move the unzipped folder `OhioT1DM` into `data`.

To train and test for the *OhioT1DM* dataset, with our optimal configuration, run
```
>> cd accurate_bg
>> python3 ohio_man.py --epoch 150
```
the default prediction horizon is 6, equivalently 30 min. To adjust to 1hr, modify
the last line of command with
```
>> python3 ohio_man.py --epoch 150 --prediction_horizon 12
```
