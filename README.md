# python-wider_eval
Python scripts for WiderFace Evaluation. 

## Introduction:
This script converts the official evaluation scripts on WiderFace from MATLAB to Python. 
Most variable and function names are reused from the MATLAB scripts to make things easier to understand. 
Also, some logics are simplified to speed up the inference time. 
The evaluation script can be run in both Python2 and Python3. 

## Usage:
**1. Clone the repository. **
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
**3. Do evaluation. **
Save the prediction inside `eval_tools/pred/` follow the required format. 
Run `python wider_eval.py` to do evaluation. 
 




