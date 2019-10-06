import sys
import unittest
import numpy as np
import pickle, gzip

class TestNetwork(unittest.TestCase):
    def __init__(self, test, netw_ctor):
        super(TestNetwork, self).__init__(test)
        self.netw_ctor = netw_ctor

    def setUp(self):
        f = gzip.open('./data/tinyTOY.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.train, self.test = u.load()

    def TestBackPropWithoutRegularization(self):
        # =====================================================
        # BackProp test without regularization
        # =====================================================
        np.random.seed(1234)
        nn_noreg = self.netw_ctor([2,3,2])
        nn_noreg.SGD_train(self.train, epochs=5, eta=0.25, lam=0.0, verbose=False)

        self.assertAlmostEqual(nn_noreg.weights[0][0,0], 1.1273524638442189)
        self.assertAlmostEqual(nn_noreg.weights[0][0,1], 0.73193288623969166)
        self.assertAlmostEqual(nn_noreg.weights[0][1,0], -0.5195223324824767)
        self.assertAlmostEqual(nn_noreg.weights[0][1,1], -0.16092443719267965)
        self.assertAlmostEqual(nn_noreg.weights[0][2,0], -1.3585942451104391)
        self.assertAlmostEqual(nn_noreg.weights[0][2,1], 1.2061099232068802)

        self.assertAlmostEqual(nn_noreg.biases[0][0][0], 0.74862652735377944)
        self.assertAlmostEqual(nn_noreg.biases[0][1][0], -1.1461180927834171)
        self.assertAlmostEqual(nn_noreg.biases[0][2][0], 0.5440188204156966)

        self.assertAlmostEqual(nn_noreg.weights[1][0,0], 0.9903972225237162)
        self.assertAlmostEqual(nn_noreg.weights[1][1,0], -0.092195162733213751)
        self.assertAlmostEqual(nn_noreg.weights[1][0,1], 0.98122432833675222)
        self.assertAlmostEqual(nn_noreg.weights[1][1,1], 0.2374002896289274)
        self.assertAlmostEqual(nn_noreg.weights[1][0,2], -1.3028262866313183)
        self.assertAlmostEqual(nn_noreg.weights[1][1,2], 0.49036035842372594)

        self.assertAlmostEqual(nn_noreg.biases[1][0][0], -0.0978044220637386)
        self.assertAlmostEqual(nn_noreg.biases[1][1][0], -0.25298179466851223)

    def TestBackPropWithRegularization(self):
        # =====================================================
        # BackProp test with regularization
        # =====================================================
        nn_reg = self.netw_ctor([2,3,2])
        nn_reg.SGD_train(self.train, epochs=5, eta=0.25, lam=0.2, verbose=False)

        self.assertAlmostEqual(nn_reg.weights[0][0,0], 0.0023322490027254854)
        self.assertAlmostEqual(nn_reg.weights[0][0,1], 0.00094433912729247342)
        self.assertAlmostEqual(nn_reg.weights[0][1,0], 0.0025152220763984155)
        self.assertAlmostEqual(nn_reg.weights[0][1,1], 0.0010184838227073763)
        self.assertAlmostEqual(nn_reg.weights[0][2,0], 0.0014913503824642196)
        self.assertAlmostEqual(nn_reg.weights[0][2,1], 0.00060372407945523312)

        self.assertAlmostEqual(nn_reg.biases[0][0][0], 0.22180045340307644)
        self.assertAlmostEqual(nn_reg.biases[0][1][0], 0.7585097820793677)
        self.assertAlmostEqual(nn_reg.biases[0][2][0], -0.51429045271149121)

        self.assertAlmostEqual(nn_reg.weights[1][0,0], -0.033370023956083815)
        self.assertAlmostEqual(nn_reg.weights[1][1,0], 0.033367780844884856)
        self.assertAlmostEqual(nn_reg.weights[1][0,1], -0.04093765557203885)
        self.assertAlmostEqual(nn_reg.weights[1][1,1], 0.040934904514374378)
        self.assertAlmostEqual(nn_reg.weights[1][0,2], -0.022491021177273487)
        self.assertAlmostEqual(nn_reg.weights[1][1,2], 0.022489509526011812)

        self.assertAlmostEqual(nn_reg.biases[1][0][0], -0.055260479604691159)
        self.assertAlmostEqual(nn_reg.biases[1][1][0], 0.055274680302746619)

def run_test_suite(name, ctor):
    if name == "prob 3":
        prob3 = unittest.TestSuite()
        for test in ["TestBackPropWithoutRegularization", "TestBackPropWithRegularization"]:
            prob3.addTest(TestNetwork(test, ctor))
        runner = unittest.TextTestRunner(verbosity=2).run(prob3).wasSuccessful(), "one or more tests for prob 3 failed"
    else:
        raise Exception('unrecognized test suite name: {}'.format(name))
