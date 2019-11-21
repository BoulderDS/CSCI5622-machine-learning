import unittest
import numpy as np


class TestPCA(unittest.TestCase):
    def __init__(self, test, pca_ctor):
        super(TestPCA, self).__init__(test)
        self.pca = pca_ctor(0.99)

    def setUp(self):
        self.X_train = np.array([[72,  4, 24, 70],[41, 43, 80, 78],[62, 19, 64, 85], [15, 45, 41, 33],[35,  6, 31, 82]])

    def TestMeanShape(self):
        self.assertEqual(self.pca.compute_mean_vector(self.X_train).shape,(4,))

    def TestCovShape(self):
        self.assertEqual(self.pca.compute_cov(self.X_train,self.pca.compute_mean_vector(self.X_train)).shape,(4,4))

    def TestReducedShape(self):
        # depending on how the target explained variance is interpreted, we can end up with
        # either 2 or 3 components; so we accept both as correct
        expected_components = [2, 3]
        ret = self.pca.fit(self.X_train)
        self.assertEqual(ret.shape[0], self.X_train.shape[0])
        self.assertIn(ret.shape[1], expected_components)

    def TestExplainedVariance(self):
        eigen_vals = [2.84574523, 1.72946803, 0.41785852, 0.00692822]
        actual = [0.5691490465397107, 0.34589360551199344, 0.083571703784301, 0.0013856441639950483]
        returned = self.pca.compute_explained_variance(eigen_vals)
        np.testing.assert_almost_equal(returned, actual, 5)

def run_test_suite(name, ctor):
    if name == "prob 1":
        prob1 = unittest.TestSuite()
        for test in ["TestMeanShape", "TestCovShape", "TestReducedShape", "TestExplainedVariance"]:
            prob1.addTest(TestPCA(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(prob1).wasSuccessful(), "one or more tests for prob 1 failed"
    else:
        raise Exception('unrecognized test suite name: {}'.format(name))
