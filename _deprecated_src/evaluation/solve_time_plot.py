import os

import pandas

from experiments.common.redis import RedisCache
from utils.setup_logger import setup_logger


def eval_solve_time(args, log_base):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    cache = RedisCache()
    logger = setup_logger("BudgetSweep", os.path.join(log_base, 'eval_solve_time'))
    patterns = [
        ("ResNet50", "ResNet50/v1/bs1/defaultshape/OPTIMAL_ILP_GC/v1/*"),
        ("VGG16", "VGG16/v1/bs1/defaultshape/OPTIMAL_ILP_GC/v1/*"),
    ]

    data = []
    for model_name, key_pattern in patterns:
        logger.info(f"Fetching {model_name} from redis with {key_pattern}")
        results, _keys = cache.query_results(key_pattern)
        total_times = [r.solve_time_s for r in results]
        for time in total_times:
            data.append({'Model': model_name, 'Solve time': float(time)})

    df = pandas.DataFrame(data)
    sns.boxplot(data=df, x='model', y='solve_time')
    plt.show()