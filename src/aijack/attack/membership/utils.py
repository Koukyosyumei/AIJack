import numpy as np


class ShadowModels:
    """Train shadow models for membership inference
       reference https://arxiv.org/abs/1610.05820

    Args
        models : torch models for shadow
    """

    def __init__(
        self,
        models,
    ):
        self.models = models
        self.num_models = len(models)

    def fit_transform(self, X, y):
        """Trains shadow models and get prediction of shadow models
           and membership label of each prediction for each class

        Args:
            X (np.array): target data
            y (np.array): target label
        Returns:
            result_dict (dict) : key is each class
                                 value is (shadow_data, shadow_label)
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        splitted_indices = np.array_split(indices, self.num_models)

        self._fit(X, y, splitted_indices)
        (
            shadow_in_preds,
            shadow_out_preds,
            shadow_in_labels,
            shadow_out_labels,
        ) = self._transform(X, y, splitted_indices)

        shadow_preds = np.concatenate([shadow_in_preds, shadow_out_preds])
        shadow_labels = np.concatenate([shadow_in_labels, shadow_out_labels])

        dummy_shadow_in_labels = np.ones_like(shadow_in_labels)
        dummy_shadow_out_labels = np.zeros_like(shadow_out_labels)
        dummy_shadow_labels = np.concatenate(
            [dummy_shadow_in_labels, dummy_shadow_out_labels]
        )

        result_dict = {}
        unique_classes = np.unique(shadow_labels)
        for c in unique_classes:
            idx = np.where(shadow_labels == c)
            result_dict[c] = (
                shadow_preds[idx],
                dummy_shadow_labels[idx],
            )

        return result_dict

    def _fit(self, X, y, splitted_indices):
        """Trains shadow models on given data

        Args:
            X (np.array): training data for shadow models
            y (np.array): training label for shadow models
        """
        for model_idx in range(self.num_models):
            self.models[model_idx].fit(
                X[splitted_indices[model_idx]], y[splitted_indices[model_idx]]
            )

    def _transform(self, X, y, splitted_indices):
        """Gets prediction and its membership label per each class
        from shadow models
        """

        shadow_in_preds = []
        shadow_out_preds = []
        shadow_in_labels = []
        shadow_out_labels = []

        for model_idx in range(self.num_models):
            in_preds = self.models[model_idx].predict_proba(
                X[splitted_indices[model_idx]]
            )
            in_labels = y[splitted_indices[model_idx]]
            shadow_in_preds.append(in_preds)
            shadow_in_labels.append(in_labels)

            out_preds = self.models[model_idx].predict_proba(
                np.delete(X, splitted_indices[model_idx], axis=0)
            )
            out_labels = np.delete(y, splitted_indices[model_idx])
            shadow_out_preds.append(out_preds)
            shadow_out_labels.append(out_labels)

        shadow_in_preds = np.concatenate(shadow_in_preds)
        shadow_out_preds = np.concatenate(shadow_out_preds)
        shadow_in_labels = np.concatenate(shadow_in_labels)
        shadow_out_labels = np.concatenate(shadow_out_labels)

        return shadow_in_preds, shadow_out_preds, shadow_in_labels, shadow_out_labels


class AttackerModel:
    def __init__(self, models):
        """Attack model for memnership inference proposed in https://arxiv.org/abs/1610.05820
        Args:
            models: models of attacker

        Attriutes:
            _init_models
            models
        """
        self.models = models

    def fit(self, shadow_result):
        """train an attacl model with the result of shadow models

        Args:
            shadow_result (dict): key is each class
                                  value is (shadow_data, shadow_label)
        """
        for label, (X, y) in shadow_result.items():
            self.models[label].fit(X, y)

    def predict(self, y_pred_prob, y_labels):
        """predict whether the given prediction came from training data or not

        Args:
            y_pred_prob (torch.Tensor): predicted probabilities on the data
            y_labels (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            in_out_pred (np.array) : result of attack
                                     each element should be one or zero
        """
        y_pred_prob = np.array(y_pred_prob)
        y_labels = np.array(y_labels)
        unique_labels = np.unique(y_labels)
        in_out_pred = np.zeros_like(y_labels)
        for label in unique_labels:
            idx = np.where(y_labels == label)
            in_out_pred[idx] = self.models[label].predict(y_pred_prob[idx])

        return in_out_pred

    def predict_proba(self, y_pred_prob, y_labels):
        """get probabilities of whether the given prediction came from
           training data or not

        Args:
            y_pred_prob (torch.Tensor): predicted probabilities on the data
            y_labels (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            in_out_pred (np.array) : result of attack
                                     each element expresses the possibility
        """
        y_pred_prob = np.array(y_pred_prob)
        y_labels = np.array(y_labels)
        unique_labels = np.unique(y_labels)
        in_out_pred = np.zeros((y_labels.shape[0], 2), dtype=float)
        for label in unique_labels:
            idx = np.where(y_labels == label)[0]
            in_out_pred[idx] = self.models[label].predict_proba(y_pred_prob[idx])
        return in_out_pred
