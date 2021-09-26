"""
Apprentissage et factorisation matricielle
M2 - MLDS - Formation Altérnance
Année : 2020/2021
Étudiants :
* ISSELNANE Hacene
* HADDAD Ayale
"""

import numpy as np
from six.moves import range
import numbers


class SVDpp():

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_yj=None, random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, trainset):
        print("Estimating biases using sgd...")
        self.trainset = trainset
        self.bu = self.bi = None
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        lr_yj = self.lr_yj

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_yj = self.reg_yj

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        rng = self.get_rng(self.random_state)

        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        yj = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors, np.double)
        global_mean = self.trainset.global_mean

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu

                # compute current error
                dot = 0
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                         reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu -
                                             reg_yj * yj[j, f])
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):
        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est

    def get_rng(random_state):
        if random_state is None:
            return np.random.mtrand._rand
        elif isinstance(random_state, (numbers.Integral, np.integer)):
            return np.random.RandomState(random_state)
        if isinstance(random_state, np.random.RandomState):
            return random_state
        raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                         'RandomState instance, got a '
                         '{}'.format(type(random_state)))

    def rmse(predictions):
        if not predictions:
            raise ValueError('Prediction list is empty.')

        mse = np.mean([float((true_r - est) ** 2)
                       for (_, _, true_r, est, _) in predictions])
        rmse_ = np.sqrt(mse)
        print('RMSE: {0:1.4f}'.format(rmse_))
        return rmse_
