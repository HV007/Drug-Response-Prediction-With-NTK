import numpy as np
import NTK
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
import subprocess

linear_svm = []
linear_regression = []
ntk_svm = []
ntk_regression = []
baysian_regression = []

max_iter = 50

for i in range(max_iter):
    print('Iteraltion: ', i+1)
    subprocess.run(['python3', 'process.py', '--number', '0', '--use_old', '0'])
    MAX_DEP = 5
    DEP_LIST = list(range(MAX_DEP))
    C_LIST = [10.0 ** i for i in range(-2, 2)]

    data_train = np.load('train.npy')
    data_test = np.load('test.npy')
    X_train = data_train[:,:-1]
    X_test = data_test[:,:-1]
    y_train = data_train[:,-1]
    y_test = data_test[:,-1]

    best_mse = 10
    best_value = 0
    for value in C_LIST:
        clf = SVR(kernel = "linear", C = value, cache_size = 100000)
        clf.fit(X_train, y_train)
        z = clf.predict(X_test)
        mse = (np.square(z-y_test)).mean()
        if mse < best_mse:
            best_mse = mse
            best_value = value
    print('MSE with linear kernel SVM: ', mse)
    linear_svm.append(mse)

    clf = KernelRidge(kernel = "linear", alpha = 1.0)
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    mse = (np.square(z-y_test)).mean()
    print('MSE with linear kernel regression: ', mse)
    linear_regression.append(mse)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    best_mse = 10
    best_value = 0
    best_dep = 0
    Ks = NTK.kernel_value_batch(X, MAX_DEP)
    a = X_train.shape[0]
    train_fold = list(range(X_train.shape[0]))
    val_fold = list(range(X_train.shape[0], X.shape[0]))

    for dep in DEP_LIST:
        for fix_dep in range(dep + 1):
            K = Ks[dep][fix_dep]
            for value in C_LIST:
                kernel_train = K[train_fold][:, train_fold]
                kernel_test = K[val_fold][:, train_fold]
                clf = SVR(kernel = "precomputed", C = value, cache_size = 100000)
                clf.fit(kernel_train, y[train_fold])
                z = clf.predict(kernel_test)
                mse = (np.square(z-y[val_fold])).mean()
                if mse < best_mse:
                    best_mse = mse
                    best_value = value
                    best_dep = dep
                    best_fix = fix_dep

    print('MSE with NTK SVM: ', best_mse)
    ntk_svm.append(best_mse)

    best_mse = 10
    best_value = 0

    for dep in DEP_LIST:
        for fix_dep in range(dep + 1):
            K = Ks[dep][fix_dep]
            kernel_train = K[train_fold][:, train_fold]
            kernel_test = K[val_fold][:, train_fold]
            clf = KernelRidge(kernel = "precomputed", alpha = 1.0)
            clf.fit(kernel_train, y[train_fold])
            z = clf.predict(kernel_test)
            mse = (np.square(z-y[val_fold])).mean()
            if mse < best_mse:
                best_mse = mse
                best_value = value
                best_dep = dep
                best_fix = fix_dep

    print('MSE with NTK regression: ', best_mse)
    ntk_regression.append(best_mse)

    clf = BayesianRidge()
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    mse = (np.square(z-y[val_fold])).mean()

    print('MSE with Baysian Regression: ', mse)
    baysian_regression.append(mse)
    print('\n')

print('\n')

linear_svm = np.array(linear_svm)
linear_regression = np.array(linear_regression)
ntk_svm = np.array(ntk_svm)
ntk_regression = np.array(ntk_regression)
baysian_regression = np.array(baysian_regression)

print('Linear Kernel SVM: ' + str(linear_svm.mean()) + "+-" + str(linear_svm.std()))
print('Linear Kernel Regression: ' + str(linear_regression.mean()) + "+-" + str(linear_regression.std()))
print('NTK SVM: ' + str(ntk_svm.mean()) + "+-" + str(ntk_svm.std()))
print('NTK Regression: ' + str(ntk_regression.mean()) + "+-" + str(ntk_regression.std()))
print('Baysian Regression: ' + str(baysian_regression.mean()) + "+-" + str(baysian_regression.std()))