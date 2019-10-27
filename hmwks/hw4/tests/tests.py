import unittest
import numpy as np


class TestSVM(unittest.TestCase):
    def __init__(self, test, svm_ctor):
        super(TestSVM, self).__init__(test)
        self.svm_ctor = svm_ctor()

    def setUp(self):
        self.sep_x = self.svm_ctor.kSEP[:, 0:2]
        self.sep_y = self.svm_ctor.kSEP[:, 2]
        self.insep_x = self.svm_ctor.kINSP[:, 0:2]
        self.insep_y = self.svm_ctor.kINSP[:, 2]

    def TestWideSlack(self):
        w = np.array([-.25, .25])
        b = -.25
        self.assertEqual(self.svm_ctor.find_slack(self.insep_x, self.insep_y, w, b),
                         set([6, 4]))

    def TestNarrowSlack(self):
        w = np.array([0, 2])
        b = -5

        self.assertEqual(self.svm_ctor.find_slack(self.insep_x, self.insep_y, w, b),
                         set([3, 5]))

    def TestSupport(self):
        w = np.array([0.2, 0.8])
        b = -0.2

        self.assertEqual(self.svm_ctor.find_support(self.sep_x, self.sep_y, w, b),
                         set([0, 4, 2]))

    def TestWeight(self):
        alpha = np.zeros(len(self.sep_x))
        alpha[4] = 0.34
        alpha[0] = 0.12
        alpha[2] = 0.22

        w = self.svm_ctor.weight_vector(self.sep_x, self.sep_y, alpha)
        self.assertAlmostEqual(w[0], 0.2)
        self.assertAlmostEqual(w[1], 0.8)

def run_test_suite(name, ctor):
    if name == "prob 1":
        prob1 = unittest.TestSuite()
        for test in ["TestWideSlack", "TestNarrowSlack", "TestSupport", "TestWeight"]:
            prob1.addTest(TestSVM(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(prob1).wasSuccessful(), "one or more tests for prob 1 failed"
    else:
        raise Exception('unrecognized test suite name: {}'.format(name))
