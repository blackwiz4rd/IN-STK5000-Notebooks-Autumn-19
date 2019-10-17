import numpy
import unittest

class CoinTossPriv(object):
    
    def __init__(self, theta):
        self.theta = theta
    
    def anonymize_one(self, x):
        if numpy.random.uniform() <= self.theta:
            return bool(x)
        else:
            return numpy.random.uniform() < self.theta
    
    def anonymize(self, data):
        pass

def split_strings(s):
    if not s:
        return []
    return [s]

class TestSplitStrings(unittest.TestCase):
    
    def test_empty(self):
        self.assertEqual(split_strings(""), [])

    def test_one_char(self):
        self.assertEqual(split_strings("a"), ["a"])

        
class TestCointToss(unittest.TestCase):
    
    def test_anonymize_one(self):
        ct_zero = CoinTossPriv(0)
        self.assertFalse(ct_zero.anonymize_one(1))
        ct_one = CoinTossPriv(1)
        self.assertTrue(ct_one.anonymize_one(1))
        self.assertFalse(ct_one.anonymize_one(0))
        
if __name__ == "__main__":
    unittest.main()