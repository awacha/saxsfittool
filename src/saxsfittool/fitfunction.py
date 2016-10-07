class FitFunction(object):
    """General class for a fit function"""

    arguments = []

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def function(self, x, *args, **kwargs):
        raise NotImplementedError

    def jacobian(self, x, *args, **kwargs):
        return None

    def initialize_arguments(self, x, y):
        return [1.0]*len(self.arguments)
