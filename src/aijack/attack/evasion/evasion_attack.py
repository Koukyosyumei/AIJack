import copy

import numpy as np
import sklearn

from ..base_attack import BaseAttacker


class Evasion_attack_sklearn(BaseAttacker):
    """Create an adversarial example against sklearn objects
        reference https://arxiv.org/abs/1708.06131

    Args:
        target_model (sklearn): sklearn classifier
        X_minus_1 (numpy.array): datasets that contains
                                only the class you want to misclasssify
        dmax (float): max distance between the adversarial example
                        and initial one
        max_iter (int): maxium number of iterations
        gamma (float): parameter gamma of svm (used for only svm)
        lam (float): trade - off parameter
        t (float): step_size
        h (float): a badwidtch paramter for a KDE
        distance (str): type of distance such as L2 or L1
        kde_type (str): type of kernel density estimator

    Attributes:
        target_model (sklearn): sklearn classifier
        X_minus_1 (numpy.array): datasets that contains
                                only the class you want to misclasssify
        dmax (float): max distance between the adversarial example
                        and initial one
        max_iter (int): maxium number of iterations
        gamma (float): parameter gamma of svm (used for only svm)
        lam (float): trade - off parameter
        t (float): step_size
        h (float): a badwidtch paramter for a KDE
        distance (str): type of distance such as L2 or L1
        kde_type (str): type of kernel density estimator
        n_minus_1 (int): number of rows of X_minus_1
        delta_g (func): deviation of he discriminant function of a
                        surrogate classifier f learnt on D

    Raises:
        ValueError: if given distance is not supported.

    Examples:
        >>>X_minus_1 = X_train[np.where(y_train == "3")]
        >>>attacker = Attack_sklearn(target_model = target_model,
                                    X_minus_1 = X_minus_1,
                                    dmax =  (5000 / 255) * 2.5,
                                    max_iter = 300,
                                    gamma = 1 / (X_train.shape[1] *
                                                np.var(X_train)),
                                    lam = 10, t = 0.5, h = 10)
        >>>xm, log = attacker.attack(x0)
    """

    def __init__(
        self,
        target_model,
        X_minus_1,
        dmax,
        max_iter,
        gamma,
        lam,
        t,
        h,
        distance="L1",
        kde_type="L1",
    ):
        super().__init__(target_model)

        self.X_minus_1 = X_minus_1
        self.dmax = dmax
        self.max_iter = max_iter
        self.gamma = gamma
        self.lam = lam
        self.t = t
        self.h = h
        self.kde_type = kde_type

        self.n_minus_1 = X_minus_1.shape[0]

        self.delta_g = None
        self.distance = None

        _ = self._detect_type_of_classifier()

        if distance == "L1":
            self.distance = lambda x1, x2: np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"distance type {distance} is not defined")

    def _detect_type_of_classifier(self):
        """set proper attributes based on the type of classifier

        Returns:
            return True (bool) if there is no error

        Raises:
            ValueError : if target classifier is not supported
        """

        target_type = type(self.target_model)

        if target_type == sklearn.svm._classes.SVC:
            params = self.target_model.get_params()
            kernel_type = params["kernel"]
            if kernel_type == "rbf":

                def kernel(xm):
                    return np.exp(
                        -self.gamma * ((xm - self.target_model.support_vectors_) ** 2)
                    )

                def delta_kernel(xm):
                    return (
                        (-2 * self.gamma)
                        * kernel(xm)
                        * (xm - self.target_model.support_vectors_)
                    )

            elif kernel_type == "linear":

                def delta_kernel(xm):
                    return self.target_model.support_vectors_

            elif kernel_type == "poly":
                p = params["degree"]
                c = self.target_model.intercept_

                def delta_kernel(xm):
                    return (
                        p
                        * (((xm * self.target_model.support_vectors_) + c) ** (p - 1))
                        * self.target_model.support_vectors_
                    )

            else:
                raise ValueError(f"kernel type {kernel_type} is not supported")

            self.delta_g = lambda xm: self.target_model.dual_coef_.dot(delta_kernel(xm))

        else:
            raise ValueError(f"target type {target_type} is not supported")

        return True

    def _get_delta_p(self, xm):
        """culculate deviation of the estimated density p(xm−1 |yc = −1)

        Args:
            xm (np.array) : an adversarial example

        Returns:
            delta_p (np.array) : deviation of p

        """

        if self.kde_type == "L1":
            a = -1 / (self.n_minus_1 * self.h)
            b = np.exp(-(np.sum(np.abs(xm - self.X_minus_1), axis=1)) / self.h).dot(
                xm - self.X_minus_1
            )
            delta_p = a * b
            return delta_p

    def _get_grad_f(self, xm, norm="l1"):
        """culculate deviation of objective function F

        Args:
            xm (np.array) : an adversarial example
            norm (str) : type of distance for normalization

        Returns:
            delta_f (np.array) : deviation of F

        Raises:
            ValueError : if the type of norm is not supported
        """

        delta_f = self.delta_g(xm) - self.lam * self._get_delta_p(xm)
        if norm == "l1":
            delta_f /= np.sum(np.abs(delta_f)) + 1e-5
        elif norm == "l2":
            delta_f /= np.sqrt(np.sum(delta_f**2, axis=0)) + 1e-5
        else:
            raise ValueError(f"norm type {norm} is not defined")

        return delta_f

    def attack(self, x0):
        """try evasion attack

        Args:
            x0 (np.array) : initial data point

        Returns:
            xm (np.array) : created adversarial example
            g_list (list) : lof of decision function (only for svm)
                            (need future improvement)
        """

        g_list = []
        xm = copy.copy(x0)
        for i in range(self.max_iter):
            delta_f = self._get_grad_f(xm)
            xm -= self.t * delta_f.reshape(-1)
            d = self.distance(xm, x0)  # + i * (10/255)
            if d > self.dmax:
                xm = x0 + ((xm - x0) / d) * self.dmax

            g_list.append(self.target_model.decision_function(xm.reshape(1, -1)))

        return xm, g_list
