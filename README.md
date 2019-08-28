# I-Keyboard: Fully Imaginary Keyboard on Touch Devices Empowered by Deep Neural Decoder
We propose a fully imaginary keyboard (I-Keyboard) with a deep neural decoder (DND). Below are a few features of I-Keyboard
- The eyes-free ten-finger typing scenario of I-Keyboard does not necessitate both a calibration step and a pre-defined region for typing (first explored in this study!).
- The invisibility of I-Keyboard maximizes the usability of mobile devices.
- DND empowered by a deep neural architecture allows users to start typing from any position on the touch screens at any angle.
- We collected the largest user data in the process of developing I-Keyboard and make the data public!
- I-Keyboard showed 18.95% and 4.06% increases in typing speed (45.57 WPM) and accuracy (95.84%), respectively over the baseline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

* CUDA >= 9.0
* Python 3.6+
* [Pytorch 1.0.0 from a nightly release](https://pytorch.org/get-started/previous-versions/)

## Data Visualization
To visualize the user behavior and analyze the statistics of user behavior, run the below.
Running below will create two directories ('list' and 'figs') and save both the preprocessed results and the figures.

```bash
cd user_behavior_analysis

# preprocess the raw_data and save the result in the 'list' directory
python3 preprocessing.py

# visualize each experiment participant's typing behavior
# and extracts the statistics over the whole participants
python3 user_analysis.py
```


## Citations

Please consider citing this project in your publications if you find this helpful.
The following is the BibTeX.

```
@article{kim2019keyboard,
  title={I-Keyboard: Fully Imaginary Keyboard on Touch Devices Empowered by Deep Neural Decoder},
  author={Kim, Ue-Hwan and Yoo, Sahng-Min and Kim, Jong-Hwan},
  journal={arXiv preprint arXiv:1907.13285},
  year={2019}
}
```