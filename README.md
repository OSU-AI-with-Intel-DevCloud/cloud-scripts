# Intel DevCloud Cloud-Scripts Setup Instructions
This repository holds the scripts used to apply Deepfake Detection inference on transfered video files from our [User Interface](https://github.com/OSU-AI-with-Intel-DevCloud/reactapp-ui).\
Follow the instructions below to correctly setup these scripts to be used in conjuncture with the User Interface.
## Hardware Specification
Everything here is done on Intel DevCloud so there are no hardware requirements.
## Setup
1. Create an Intel DevCloud oneAPI account for free at: [devcloud.intel.com/oneapi/get_started/](https://devcloud.intel.com/oneapi/get_started/).
2. Connect to Intel DevCloud either through JupyterLabs or through ssh.
3. Create a new directory labeled `deepfake` and `cd ./deepfake`.
4. In this directory `git clone https://github.com/OSU-AI-with-Intel-DevCloud/cloud-scripts`.
5. Switch folders to /input and `mkdir combined`. Move back to /deepfake.
6. Install the necessary software to run the python script: torch, torchvision, opencv, matlib and more depending on your machine.
(There will be an indication of the necesary software to install when running the script)
7. Run `python combinedlistener.py`.
This script will continously wait for a file to be transferred into /input/combined. Once it detects a file here it will perform inference on the video file to test
if it is a deepfake or not. The file will be deleted from the folder and the results of the inference will be output to /output/submission.csv.
The script will continue waiting for more input until you manually end the script with CTRL+C. If there are issues related to the MOV file this means that there are
timing issues and the script is attempting to read the inputted file to quickly. In this scenario make sure to increase the wait time before file read. Similarly
match this timing with some space in the backend which has to wait for results to read.

## Further Improvement
This area of the project has massive room for growth. By using Intel DevCloud and Jupyter Notebooks
a user can quickly explore a variety of shared Deepfake Detection [solutions](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion?sort=votes)
and [code](https://www.kaggle.com/competitions/deepfake-detection-challenge/code?competitionId=16880&sortBy=scoreAscending) from the Kaggle Deepfake Detection Challenge in depth.
Our project mainly looked through two of the shared algorithms and tested how well they performed through Intel DevCloud in order to learn and explore.
However, I suggest future teams interested in this project to go beyond and model and train their own algorithms using insights from Kaggle's vast open source resources.
