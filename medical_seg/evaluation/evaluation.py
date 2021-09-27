from collections import OrderedDict
from .general import ConfusionMatrix, ALL_METRICS
import numpy as np 

class Metric:

    def __init__(self, class_name=[], voxel_spacing=(1, 1, 1), nan_for_nonexisting=True) -> None:
        ## 不计算背景，因此把从1开始
        assert len(class_name) != 0, "请传入class_name参数。"
        self.class_num = len(class_name) + 1
        self.nan_for_nonexisting = nan_for_nonexisting
        self.class_name = class_name
        self.voxel_spacing = voxel_spacing
        
    def run(self, pred, label):
        # pred为argmax以后的结果
        res = OrderedDict({f"{c}_{m}": 0.0 for c in self.class_name for m in ALL_METRICS})

        for c in range(self.class_num):
            if c == 0:
                continue
            c_name = self.class_name[c - 1]
            for m_name, m_func in ALL_METRICS.items():
                res[f"{c_name}_{m_name}"] = m_func(test=(pred == c), reference=(label == c), 
                                                    confusion_matrix=None, nan_for_nonexisting=self.nan_for_nonexisting, 
                                                    voxel_spacing=self.voxel_spacing)
        return res

if __name__ == "__main__":
    m = Metric(class_name=["1", "2", "3"])

    a1 = np.random.rand(1, 32, 32) > 0.5
    a2 = np.random.rand(1, 32, 32) > 0.5

    print(m.run(a1, a2))