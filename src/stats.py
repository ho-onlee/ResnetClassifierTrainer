import os
import dill
from dynaconf import Dynaconf
import matplotlib.pyplot as plt
import logging

settings = Dynaconf(settings_files=["../settings.toml"])

logger = logging.getLogger(__name__)

def datasetStats():
    datas = os.listdir(os.path.join(settings.folders.data_root, 'pre_processed_entries'))
    labels = []
    length = []
    for sample in datas:
        with open(os.path.join(settings.folders.data_root, 'pre_processed_entries', sample), 'rb') as f:
            entry = dill.load(f)
            labels.append(entry['label'])
            length.append(entry['end'] - entry['start'])
    
    from collections import Counter

    counts = Counter(labels)
    summa = []

    for item, count in counts.items():
        percentage = count / len(labels) * 100
        logger.info(f"{item}: {count} ({percentage:.2f}%)")
        summa.append((item, count, percentage))

    print("\nClass Distribution:")
    for item, count, percentage in summa:
        print(f"{item}: {count} ({percentage:.2f}%)")
    plt.bar([x[0] for x in summa], [x[1] for x in summa])
    plt.savefig(os.path.join(settings.folders.data_root, 'class_distribution.png'))
    plt.show()

