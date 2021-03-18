class GenericParamControl(object):
    @staticmethod
    def init_module_list():
        raise NotImplementedError

    @staticmethod
    def init_param_group():
        raise NotImplementedError

    def get_param_group(self):
        raise NotImplementedError

    def release_param(self):
        raise NotImplementedError
