import numpy as np
import NTK
from sklearn.svm import SVR

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
print('MSE with linear kernel: ', mse)

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

best_mse = 10
best_value = 0
best_dep = 0
best_ker = 0
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

print('MSE with NTK: ', best_mse)