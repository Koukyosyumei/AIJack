import torch.nn.functional as F

from ..fedavg import FedAvgClient


class FedKDClient(FedAvgClient):
    """Implementation of FedKD (https://arxiv.org/abs/2108.13323)"""

    def __init__(
        self,
        student_model,
        teacher_model,
        task_lossfn,
        student_lr=0.1,
        teacher_lr=0.1,
        adaptive_distillation_losses=True,
        adaptive_hidden_losses=True,
        gradient_compression_ratio=1.0,
        user_id=0,
        send_gradient=True,
    ):
        super(FedKDClient, self).__init__(
            student_model, user_id=user_id, lr=student_lr, send_gradient=send_gradient
        )
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_lr = teacher_lr
        self.task_lossfn = task_lossfn
        self.adaptive_distillation_losses = adaptive_distillation_losses
        self.adaptive_hidden_losses = adaptive_hidden_losses
        self.gradient_compression_ratio = gradient_compression_ratio

        self._is_valid_models()

    def _is_valid_models(self):
        if self.adaptive_hidden_losses:
            if not hasattr(self.teacher_model, "get_hidden_states"):
                raise AttributeError(
                    "If adaptive_hidden_losses=True,\
                 teacher_model must have `get_hidden_states` method"
                )
            if not hasattr(self.student_model, "get_hidden_states"):
                raise AttributeError(
                    "If adaptive_hidden_losses=True,\
                 student_model must have `get_hidden_states` method"
                )

    def loss(self, x, y):
        y_pred_teacher = self.teacher_model(x)
        y_pred_student = self.student_model(x)

        teacher_loss = 0
        student_loss = 0

        # task_losses
        task_loss_teacher = self.task_lossfn(y_pred_teacher, y)
        task_loss_student = self.task_lossfn(y_pred_student, y)
        teacher_loss += task_loss_teacher
        student_loss += task_loss_student

        # adaptive_distillation_losses
        if self.adaptive_distillation_losses:
            adaptive_distillaion_loss_teacher = F.kl_div(
                y_pred_student.softmax(dim=1).log(), y_pred_teacher.softmax(dim=1)
            ) / (task_loss_student + task_loss_teacher)
            adaptive_distillaion_loss_student = F.kl_div(
                y_pred_teacher.softmax(dim=1).log(), y_pred_student.softmax(dim=1)
            ) / (task_loss_student + task_loss_teacher)
            teacher_loss += adaptive_distillaion_loss_teacher
            student_loss += adaptive_distillaion_loss_student

        # adaptove_hidden_losses
        if self.adaptive_hidden_losses:
            adaptive_hidden_losses_student_teacher = 0
            hidden_states_teacher = self.teacher_model.get_hidden_states()
            hidden_states_student = self.student_model.get_hidden_states()

            if type(hidden_states_teacher) != list:
                raise TypeError(
                    "get_hidden_states should return a list of torch.Tensors"
                )
            if type(hidden_states_student) != list:
                raise TypeError(
                    "get_hidden_states should return a list of torch.Tensors"
                )

            for hst, hss in zip(hidden_states_teacher, hidden_states_student):
                adaptive_hidden_losses_student_teacher += F.mse_loss(hst, hss)
            teacher_loss += adaptive_hidden_losses_student_teacher / (
                task_loss_student + task_loss_teacher
            )
            student_loss += adaptive_hidden_losses_student_teacher / (
                task_loss_student + task_loss_teacher
            )

        return teacher_loss, student_loss
