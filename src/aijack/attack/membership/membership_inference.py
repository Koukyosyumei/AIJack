from ..base_attack import BaseAttacker
from .utils import AttackerModel, ShadowModel


class Membership_Inference(BaseAttacker):
    def __init__(
        self,
        target_model,
        shadow_models,
        attack_models,
        shadow_data_size,
        shadow_transform,
    ):
        """Implementation of membership inference
           reference https://arxiv.org/abs/1610.05820

        Args:
            target_model: the model of the victim
            shadow_models: shadow model for attack
            attack_models: attacker model for attack
            shadow_data_size: the size of datasets for
                              training the shadow models
            shadow_transform: the transformation function for shadow datasets

        Attributes:
            shadow_models
            attack_models
            shadow_data_size
            shadow_trnasform
            sm
            shadow_result
            am
        """
        super().__init__(target_model)
        self.shadow_models = shadow_models
        self.attack_models = attack_models
        self.shadow_data_size = shadow_data_size
        self.shadow_trasform = shadow_transform

        self.sm = None
        self.shadow_result = None
        self.am = None

    def train_shadow(self, X, y, num_itr):
        """train shadow models

        Args:
            X (np.array): training data for shadow models
            y (np.array): training label for shadow models
            num_itr (int): number of iteration for training
        """
        self.sm = ShadowModel(
            self.shadow_models,
            self.shadow_data_size,
            shadow_transform=self.shadow_trasform,
        )
        self.shadow_result = self.sm.fit_transform(X, y, num_itr=num_itr)

    def train_attacker(self):
        """Train attacker models"""
        self.am = AttackerModel(self.attack_models)
        self.am.fit(self.shadow_result)

    def attack(self, x, y, proba=False):
        """Attack victim model

        Args:
            x: target datasets which the attacker wants to classify
            y: target labels which the attacker wants to classify
            proba: the format of the output
        """
        prediction_of_taregt_model = self.target_model(x)
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
