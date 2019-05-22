# python-wider_eval
Python script for WIDER FACE evaluation. 

## Introduction:
This script converts the official evaluation scripts on WIDER FACE from MATLAB to Python. 
The same logic is followed with most variable and function names reused from the MATLAB scripts to make it easier to understand. 
Some logics are simplified to speed up the inference time and we observe a 2.5x speed up compared to the MATLAB scripts.
The evaluation script can be run in both Python2 and Python3. 

## Usage:
**1. Clone the repository.**
```bash
git clone https://github.com/xiyinmsu/python-wider_eval.git
cd python-wider_eval/
```
**2. Download the official evaluation scripts.**
```bash
wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
unzip eval_tools.zip
rm eval_tools.zip
```
**3. Do evaluation.**

Save the prediction results inside `eval_tools/pred/` follow the required format. 
And then run the following command for evaluation. 
You can also change the ground truth and prediction file paths if they are not the default ones. 
There is an optional parameter to select prediction results with confidence larger than a pre-defined threshold (`-s`).
For example, the following command will only select results with confidence larger than 0.1 for evaluation.
```bash
python wider_eval.py -s 0.1
```
 




