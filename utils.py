import ray

def get_cluster_resources() -> dict :
    available_resources = ray.available_resources()

    return available_resources


if __name__ == "__main__":
    resources = get_cluster_resources()
    print(f"Available resources: {resources}")