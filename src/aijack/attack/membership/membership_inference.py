from ..base_attack import BaseAttacker
from .utils import AttackerModel, ShadowModels


class Membership_Inference(BaseAttacker):
    def __init__(
        self,
        target_model,
        shadow_models,
        attack_models,
    ):
        """Implementation of membership inference
           reference https://arxiv.org/abs/1610.05820

        Args:
            target_model: the model of the victim
            shadow_models: shadow model for attack
            attack_models: attacker model for attack
        """
        super().__init__(target_model)
        self.sm = ShadowModels(shadow_models)
        self.am = AttackerModel(attack_models)
        self.shadow_result = None

    def fit(self, X, y):
        self.train_shadow(X, y)
        self.train_attacker()

    def train_shadow(self, X, y):
        """train shadow models

        Args:
            X (np.array): training data for shadow models
            y (np.array): training label for shadow models
        """
        self.shadow_result = self.sm.fit_transform(X, y)

    def train_attacker(self):
        """Train attacker models"""
        self.am.fit(self.shadow_result)

    def attack(self, x, y, proba=False):
        """Attack victim model

        Args:
            x: target datasets which the attacker wants to classify
            y: target labels which the attacker wants to classify
            proba: the format of the output
        """
        prediction_of_taregt_model = self.target_model.predict_proba(x)
        if proba:
            return self.predit_proba(prediction_of_taregt_model, y)
        else:
            return self.predit(prediction_of_taregt_model, y)

    def predict(self, pred, label):
        """Predict whether the given prediction came from training data or not

        Args:
            pred (torch.Tensor): predicted probabilities on the data
            label (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            predicted binaru labels
        """
        return self.am.predict(pred, label)

    def predict_proba(self, pred, label):
        """get probabilities of whether the given prediction came from
           training data or not

        Args:
            pred (torch.Tensor): predicted probabilities on the data
            label (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            predicted probabilities
        """
        return self.am.predict_proba(pred, label)
