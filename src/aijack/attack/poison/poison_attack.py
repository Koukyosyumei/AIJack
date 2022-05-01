import copy

import numpy as np
import sklearn
from tqdm import tqdm

from ..base_attack import BaseAttacker


class Poison_attack_sklearn(BaseAttacker):
    """implementation of poison attack for sklearn binary classifier
        reference https://arxiv.org/abs/1206.6389

    Args:
        target_model: sklean classifier
        X_train: training data for target_model
        y_train: training label for target_model
        t: step size

    Attributes:
        target_model: sklean classifier
        X_train:
        y_train:
        t: step size
        kernel
        delta_kernel
    """

    def __init__(self, target_model, X_train, y_train, t=0.5):
        super().__init__(target_model)

        self.X_train = X_train
        self.y_train = y_train
        self.t = t

        self.kernel = None
        self.delta_kernel = None

        _ = self._detect_type_of_classifier()

    def _detect_type_of_classifier(self):
        """detect the type of classifier and prepare proper settings

        Returns:
            return true if no error occurs

        Raises:
            ValueError: if given kernel type is not supported.
        """
        target_type = type(self.target_model)

        if target_type == sklearn.svm._classes.SVC:
            params = self.target_model.get_params()
            kernel_type = params["kernel"]
            if kernel_type == "linear":
                self.kernel = lambda xa, xb: xa.dot(xb.T)
                self.delta_kernel = lambda xi, xc: self.t * xi
            else:
                raise ValueError(f"kernel type {kernel_type} is not supported")

        else:
            raise ValueError(f"target type {target_type} is not supported")

        return True

    def _delta_q(self, xi, xc, yi, yc):
        """Culculate deviation of q
           Q = yy.T * K denotes the label - annotated version of K,
           and α denotes the SVM’s dual variables corresponding
           to each training point.

        Args:
            xi: intermidiate results of the generation of adversarial example
            xc: initial attack point
            yi: the labels of intermidiate results of the generation
                of adversarial example
            yc: true label of initial attack point

        Returns:
            dq:
        """
        d = xi.shape[1]
        yy = np.array([(yi * yc)] * d).T
        dq = yy * (self.delta_kernel(xi, xc))
        return dq

    def attack(self, xc, yc, X_valid, y_valid, num_iterations=200):
        """Create an adversarial example for poison attack

        Args:
            xc: initial attack point
            yc: true label of initial attack point
            X_valid: validation data for target_model
            y_valid: validation label for target_model
            num_iterations: (default = 200)

        Returns:
            xc: created adversarial example
            log: log of score of target_model under attack
        """
        # flip the class label
        yc *= -1
        log = []
        X_train_poisoned = copy.copy(self.X_train)
        y_train_poisoned = copy.copy(self.y_train)

        # best_score = float("inf")
        # best_xc = None
        # best_itr = None

        for i in tqdm(range(num_iterations)):

            target_model_ = copy.copy(self.target_model)
            target_model_.__init__()
            target_model_.set_params(**self.target_model.get_params())
            # target_model_ = sklearn.svm.SVC(kernel="linear", C=1)

            # add poinsoned data
            target_model_.fit(
                np.concatenate([X_train_poisoned, xc.reshape(1, -1)]),
                np.concatenate([y_train_poisoned, [yc]]),
            )

            score_temp = target_model_.score(X_valid, y_valid)
            log.append(score_temp)
            # if score_temp < best_score:
            #    best_score = score_temp
            #    best_xc = xc
            #    best_itr = i

            # ------------------------ #
            xs = target_model_.support_vectors_
            ys = np.concatenate([y_train_poisoned, [yc]])[target_model_.support_]

            Qks = y_valid.reshape(-1, 1).dot(ys.reshape(-1, 1).T) * self.kernel(
                X_valid, xs
            )
            Qss_inv = np.linalg.inv(self.kernel(xs, xs))
            v = Qss_inv.dot(ys)
            zeta = ys.T.dot(v)
            Mk = (-1 / zeta) * (
                (Qks).dot(zeta * Qss_inv - v.dot(v.T))
                + y_valid.reshape(-1, 1).dot(v.reshape(1, -1))
            )

            delta_Qsc = self._delta_q(xs, xc.reshape, ys, yc)
            delta_Qkc = self._delta_q(X_valid, xc.reshape(1, -1), y_valid, yc)

            # α denotes the SVM’s dual variables corresponding to each
            # training point
            alpha = target_model_.decision_function([xc])

            # the desired gradient used for optimizing our attack:
            delta_L = np.sum(((Mk.dot(delta_Qsc) + delta_Qkc) * alpha), axis=0)

            # u is a norm-1 vector representing the attack direction,
            u = delta_L / np.sqrt(np.sum((delta_L**2)))

            # the attack point
            xc += self.t * u

        # print(f"initial score is {log[0]}")
        # print(f"best score is {best_score} in iteration {best_itr}")

        return xc, log
