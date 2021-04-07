from base.parameter_control import ResnetParamControl


class ParamControl(ResnetParamControl):

    def init_module_list(self):

        module_list = [[(4, 10), (163, 187)], [(142, 163)], [(121, 142)]]
        if self.backbone_mode == "ir_se":
            module_list = [[(4, 10), (205, 235)], [(187, 205)], [(169, 187)]]

        return module_list


