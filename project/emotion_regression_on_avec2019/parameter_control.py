from base.parameter_control import GenericParamControl


class ParamControl(GenericParamControl):
    def __init__(self, trainer, release_count=8):
        self.trainer = trainer
        self.release_count = release_count
        self.module_list = self.init_module_list()
        self.module_stack = self.init_module_list()
        self.module_to_layer_mapping_index = self.init_param_group()

    @staticmethod
    def init_module_list():
        return ['0', '1', '2', '3', '4', '5', '6', '7']

    @staticmethod
    def init_param_group():
        return {'0': slice(151, 160), '1': slice(160, 169), '2': slice(169, 178),
                '3': slice(178, 187), '4': slice(187, 196), '5': slice(196, 205),
                '6': slice(205, 235), '7': slice(4, 10)}

    def get_param_group(self):
        modules_to_release = self.module_stack.pop()
        return modules_to_release

    def get_current_lr(self):
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        return current_lr

    def release_param(self):
        if self.release_count > 0:
            module = self.get_param_group()

            indices = self.module_to_layer_mapping_index[module]
            for param in list(self.trainer.model.parameters())[indices]:
                param.requires_grad = True

            self.trainer.init_optimizer_and_scheduler()
            self.release_count -= 1


