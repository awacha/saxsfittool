import numpy as np

from ..core import FitFunction

class YarussoCooper(FitFunction):
    name = "Yarusso-Cooper model for ionic aggregates"

    description = "Yarusso-Cooper model as used in Macromol. 33 (10) 2000 p. 3818"

    arguments = [('factor', 'Intensity scaling factor'),
                 ('R1','Ionic core radius'),
                 ('RCA','Hydrocarbon chain shell radius'),
                 ('vp','Mean volume of an aggregate'),
    ]

    @staticmethod
    def PHI(x):
        return 3/x**3*(np.sin(x)-x*np.cos(x))

    def function(self, x, factor, R1, RCA, vp):
        v1=4*np.pi*R1**3/3
        vCA=4*np.pi*RCA**3/3
        return factor*v1**2*self.PHI(x*R1)**2/(1+(8*vCA/vp)*self.PHI(2*x*RCA)**2)

