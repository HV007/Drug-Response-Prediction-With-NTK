# Drug Response Prediction with NTK
# File Description
1. `data.npy`: Processed Drug Response data from PDX database with 1634 examples, taken from [1]
2. `process.py`: Divides `data.npy` into train/test split in a ratio of roughly 2:1, taken from [1]
3. `NTK.py`: Neural Tangent Kernel, taken from [2]
4. `script.py`: Calculates Mean Squared Error using SVM with both linear kernel and NTK

# Execution
1. Create test/train split by running: `python3 process.py --number 0 --use_old 0`
2. Execute ths script: `python3 script.py`

# Results
1. MSE with linear kernel reported in [1]: 0.824 +- 0.034
2. MSE with linear kernel I got: 0.815 +- 0.157
3. MSE with NTK kernel I got: 0.792 +- 0.162

# References
1. [DrugOrchestra](https://github.com/jiangdada1221/DrugOrchestra)
2. [NTK](https://github.com/LeoYu/neural-tangent-kernel-UCI)