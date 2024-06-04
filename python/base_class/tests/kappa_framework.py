import unittest

import numpy as np

from ..physics.di_higgs import Coupling, ggF


class KappaFrameworkReweight(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_basis = ggF(
            Coupling(
                dict(kl=0.0, xs=0.069725),
                dict(kl=1.0, xs=0.031047),
                dict(kl=5.0, xs=0.091172),
            )
        )
        self.test_couplings = Coupling(kl=2.45)

    def test_coupling(self):
        c = Coupling(kl=[0.0, 3.0, 6.0], kt=[0.5, 3.5, 6.5])
        c.broadcast(kl=5.0, kt=[1.0, 2.0, 3.0])
        c.cartesian(kl=[0.0, 2.0], kt=[1.0, 3.0])
        self.assertTrue(
            np.all(
                c.array("kl", "kt", "kv")  # kv=default=1.0
                == [
                    # basic
                    [0.0, 0.5, 1.0],
                    [3.0, 3.5, 1.0],
                    [6.0, 6.5, 1.0],
                    # broadcasted
                    [5.0, 1.0, 1.0],
                    [5.0, 2.0, 1.0],
                    [5.0, 3.0, 1.0],
                    # cartesian
                    [0.0, 1.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 3.0, 1.0],
                ]
            )
        )

    def test_weight(self):
        self.assertTrue(
            np.all(
                np.isclose(
                    self.test_basis.weight(self.test_couplings),
                    [[-0.7395, 1.561875, 0.177625]],
                )
            )
        )

    def test_xs(self):
        # kl=2.45, xs=0.013124
        self.assertEqual(
            self.test_basis.xs(self.test_couplings),
            0.013124322125000185,
        )


if __name__ == "__main__":
    unittest.main()
