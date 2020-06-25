# I-Keyboard: Fully Imaginary Keyboard on Touch Devices Empowered by Deep Neural Decoder
We propose a fully imaginary keyboard (I-Keyboard) with a deep neural decoder (DND). Below are a few features of I-Keyboard.
- The eyes-free ten-finger typing scenario of I-Keyboard does not necessitate both a calibration step and a pre-defined region for typing (first explored in this study!).
- The invisibility of I-Keyboard maximizes the usability of mobile devices.
- DND empowered by a deep neural architecture allows users to start typing from any position on the touch screens at any angle.
- We collected the largest user data in the process of developing I-Keyboard and make the data public!
- I-Keyboard showed 18.95% and 4.06% increases in typing speed (45.57 WPM) and accuracy (95.84%), respectively over the baseline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements for Data Collection
* MS Visual Studio >= 2015

### Requirements for DND

* CUDA >= 9.0
* Python 3.6+
* Tensorflow >= 1.14

## Data Collection
### Data Format
1. File Name: We have two types of file names.
- D_A_S_I_T_P
  - D: date (YYYYMMDD).
  - A: age.
  - S: sex (male or female).
  - I: initial for discrimination.
  - T: typing speed on physical keyboard.
  - P: palm attached or detached while typing.
  - e.g. 20190117_24_male_lhk_200_x
- D_A_S_M_T
  - M: major for discrimination.
  - e.g. 20180831_26_male_enginerring_140
2. Data Format
- One or two line(s) of phrases.
  - When an enter is involved, two lines appear.
  - The two phrases are separated by the enter key.
- The sequence of x touch positions.
- The sequence of y touch positions.

## Data Visualization
To visualize the user behavior and analyze the statistics of user behavior, run the below.
Running below will create two directories ('list_data' and 'figs') and save both the preprocessed results and the figures.

```bash
cd user_behavior_analysis

# preprocess the raw_data and save the result in the 'list_data' directory
python3 preprocessing.py

# visualize each experiment participant's typing behavior
# and extracts the statistics over the whole participants
python3 user_analysis.py
```

## Training and Testing
For training, you need to set up a conda environment (recommended), or you can use pip instead
```bash
conda cread --name ikeyboard python=3.6
conda activate ikeyboard
conda install -c conda-forge tensorflow-gpu=1.14 editdistance
```

Then, you need to prepare data record using the "data.py" script as follows
```bash
python data.py
```

Finally, you can train and test the propose DND model as follows
```bash
python train.py --name experiment_name
python test_experiment.py
```
For training options and test options, refer to "train_script.py" and "test_script.py".

## Notification
- The paper has been accepted! (IEEE Trans. on Cybernetics)
- Any comments are welcome
- Thank you for your attention

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

## Acknowledgement
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2017R1A2A1A17069837).