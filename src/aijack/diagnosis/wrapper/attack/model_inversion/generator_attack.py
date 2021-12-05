from ..core import WrapperAttacker


class WrapperGeneratorAttack(WrapperAttacker):
    def __init__(
        self,
        attacker,
        attack_params={"epochs": 10},
        attacker_criterions=None,
        victim_criterions=None,
        attacker_optimizers=None,
        victim_optimizers=None,
        victim_train_dataloader=None,
        victim_test_dataloader=None,
        auxiliary_dataloader=None,
        device="cpu",
    ):
        super().__init__(
            attacker,
            attack_params,
            attacker_criterions,
            victim_criterions,
            attacker_optimizers,
            victim_optimizers,
            victim_train_dataloader,
            victim_test_dataloader,
            auxiliary_dataloader,
            device,
        )

    def pseudo_train(self, epochs):
        self.attacker.target_model.train()
        for _ in range(epochs):
            epoch_loss = 0
            epoch_outputs = []
            epoch_labels = []
            for i, data in enumerate(self.victim_train_dataloader[0]):
                for opt in self.victim_optimizers:
                    opt.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.attacker.target_model(inputs)
                loss = self.victim_criterions[0](outputs, labels)
                loss.backward()
                self.attacker.target_model.backward(outputs.grad)
                epoch_loss += loss.item() / len(self.victim_train_dataloader[0].dataset)

                epoch_outputs.append(outputs)
                epoch_labels.append(labels)

                for opt in self.victim_optimizers:
                    opt.step()

    def attack(self):
        result = {}
        self.attacker.fit(self.auxiliary_dataloaders[0], self.avg_params["epochs"])
        if self.victim_train_dataloaders is not None:
            result["reconstructed_train_data"] = (
                self.attacker.attack(self.victim_train_dataloaders[0]).detach().numpy()
            )
        if self.victim_test_dataloaders is not None:
            result["reconstructed_test_data"] = (
                self.attacker.attack(self.victim_test_dataloaders[0]).detach().numpy()
            )
        return result
