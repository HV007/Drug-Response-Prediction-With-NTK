# Drug Response Prediction with NTK
## File Description
1. `data.npy`: Processed Drug Response data from PDX database with 1634 examples, taken from [1]
2. `process.py`: Divides `data.npy` into train/test split in a ratio of roughly 2:1, taken from [1]
3. `NTK.py`: Neural Tangent Kernel, taken from [2]
4. `script.py`: Calculates Mean Squared Error using SVM with both linear kernel and NTK

## Execution
1. Execute ths script: `python3 script.py`

## Results
1. MSE with linear kernel SVM reported in [1]: 0.824 +- 0.034
2. MSE with linear kernel SVM I got: 0.807 +- 0.134
3. MSE with linear kernel regression I got: 0.817 +- 0.120
4. MSE with NTK kernel SVM I got: 0.755 +- 0.128
5. MSE with NTK kernel regression I got: 0.788 +- 0.109
6. MSE with Baysian Regression I got: 0.804 +- 0.122
7. MSE with Random Forest Regression I got: 0.813 +- 0.141

## References
1. [DrugOrchestra](https://github.com/jiangdada1221/DrugOrchestra)
2. [NTK](https://github.com/LeoYu/neural-tangent-kernel-UCI)

# Drug Task Prediction with GNN
Refer to GNN folder.
Datasets explored: Drugbank and Repurposing Hub
Earlier AUROC score: Drugbank - 0.752 and Repurposing Hub - 0.703
New AUROC score: Combined - 0.825
