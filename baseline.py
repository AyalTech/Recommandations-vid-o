"""
Apprentissage et factorisation matricielle
M2 - MLDS - Formation Altérnance
Année : 2020/2021
Étudiants :
* ISSELNANE Hacene
* HADDAD Ayale
"""

import numpy as np


class Baseline():
    def __init__(self, reg=0.02, learning_rate=.00005, n_epochs=20, verbose=False):
        self.n_epochs = n_epochs
        self.reg = reg
        self.learning_rate=learning_rate
        self.verbose = verbose

    def fit(self, trainset):
        self.trainset = trainset
        self.bu = self.bi = None
        self.bu, self.bi = self.compute_baselines()
        return self

    def estimate(self, u, i):
        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        return est

    def baseline_sgd(self):
        bu = np.zeros(self.trainset.n_users)
        bi = np.zeros(self.trainset.n_items)
        global_mean = self.trainset.global_mean

        n_epochs = self.bsl_options.get('n_epochs', 20)
        reg = self.bsl_options.get('reg', 0.02)
        lr = self.bsl_options.get('learning_rate', 0.005)

        for dummy in range(n_epochs):
            for u, i, r in self.trainset.all_ratings():
                err = (r - (global_mean + bu[u] + bi[i]))
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

        return bu, bi

    def compute_baselines(self):
        if self.bu is not None:
            return self.bu, self.bi
        self.bu, self.bi = self.baseline_sgd(self)
        return self.bu, self.bi

    def rmse(predictions):
        if not predictions:
            raise ValueError('Prediction list is empty.')

        mse = np.mean([float((true_r - est) ** 2)
                       for (_, _, true_r, est, _) in predictions])
        rmse_ = np.sqrt(mse)
        print('RMSE: {0:1.4f}'.format(rmse_))
        return rmse_