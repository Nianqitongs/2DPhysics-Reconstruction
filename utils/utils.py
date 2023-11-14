# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2023/11/14 13:12
import random
import os
import numpy as np
def seed_torch(seed=66):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)