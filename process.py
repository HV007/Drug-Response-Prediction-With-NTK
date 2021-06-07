import numpy as np
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--number',type=int,help='the index of the split of the dataset,the training and test set are stored for later use')
    parser.add_argument('--use_old',type=int,default=1,help='whether to use stored training and test set,if set to 0, will create new split of training and test')
    args = parser.parse_args()
    paths = ['data.npy']
    paths_train = ['train.npy']
    paths_test = ['test.npy']

    index = args.number
    use_old = True if args.use_old == 1 else False
    for k in range(1):
        print(k)
        data = np.load(paths[k])
        passed = False
        if not use_old:
            print('use new')
            while passed != True:
                drug = set()
                for i in range(data.shape[0]):
                    drug.add(str(list(np.round(data[i,0:20],5))))
                drug = list(drug)
                drug_train = set(list(np.random.choice(drug,len(drug)*2//3,replace=False)))
                drug_test = set()
                for i in range(len(drug)):
                    if drug[i] not in drug_train:
                        drug_test.add(drug[i])
                train_index = []
                test_index = []
                for i in range(data.shape[0]):
                    d = str(list(np.round(data[i,0:20],5)))
                    if d in drug_train:
                        train_index.append(i)
                    else :
                        assert d in drug_test
                        test_index.append(i) 
                if k != 0:
                    passed = True
                else :
                    # print(len(train_index))
                    #to avoid too many drugs with only few interactions in training
                    # if len(train_index)/256 >=2310 and len(train_index)/256<=2460: 
                    passed=True
                # if passed:
                #     np.save('index/{0}_{1}_train.npy'.format(str(index),str(k)),train_index)
                #     np.save('index/{0}_{1}_test.npy'.format(str(index),str(k)),test_index) 
        else :
            train_index = np.load('index/{0}_{1}_train.npy'.format(str(index),str(k)))
            test_index = np.load('index/{0}_{1}_test.npy'.format(str(index),str(k)))       
        np.save(paths_train[k],data[train_index,:])
        np.save(paths_test[k],data[test_index,:])

main()
