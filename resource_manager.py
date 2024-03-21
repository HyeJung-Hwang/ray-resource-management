
from utils import get_cluster_resources


def get_otimized_cpu_workers() -> int:
    available_resources = get_cluster_resources()
    cpu_cores = available_resources["CPU"]
    return int(cpu_cores/2 - 1)