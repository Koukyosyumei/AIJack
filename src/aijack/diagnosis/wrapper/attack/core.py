class WrapperAttacker:
    def __init__(
        self,
        attacker,
        attack_params={},
        attacker_criterions=[],
        victim_criterions=[],
        attacker_optimizers=[],
        victim_optimizers=[],
        victim_train_dataloaders=[],
        victim_test_dataloaders=[],
        auxiliary_dataloaders=[],
        device=None,
    ):
        self.attacker = attacker
        self.attack_params = attack_params
        self.attacker_criterions = attacker_criterions
        self.victim_criterions = victim_criterions
        self.attacker_optimizers = attacker_optimizers
        self.victim_optimizers = victim_optimizers
        self.victim_train_dataloaders = victim_train_dataloaders
        self.victim_test_dataloaders = victim_test_dataloaders
        self.auxiliary_dataloaders = auxiliary_dataloaders
        self.device = device

    def pseudo_train(self):
        pass

    def attack(self):
        pass

    def score(self):
        pass

    def save(self):
        pass
