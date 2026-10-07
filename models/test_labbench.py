import unittest
import numpy as np
from labbench import *

class TestDelayLine(unittest.TestCase):
    
    def test_delay(self):
        depth = 3
        N = 2
        delay_line = DelayLine(depth, N)
        
        # 入力信号を生成
        signal_1 = np.array([1, 2])
        signal_2 = np.array([3, 4])
        signal_3 = np.array([5, 6])

        # 最初の呼び出し: 遅延ラインはゼロなので、出力もゼロであるべき
        output = delay_line(signal_1)
        self.assertTrue(np.all(output == 0))

        output = delay_line(signal_2)
        self.assertTrue(np.all(output == 0))

        output = delay_line(signal_3)
        self.assertTrue(np.all(output == 0))

        output = delay_line(np.zeros(N))
        self.assertTrue(np.all(output == signal_1))

        output = delay_line(np.zeros(N))
        self.assertTrue(np.all(output == signal_2))
        
    def test_no_delay(self):
        N = 2
        delay_line = DelayLine(0, N)
        signal = np.array([1, 2])

        # depthが0の場合、信号がそのまま返される
        output = delay_line(signal)
        self.assertTrue(np.all(output == signal))

if __name__ == '__main__':
    unittest.main()