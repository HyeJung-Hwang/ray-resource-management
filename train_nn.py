import os
import tempfile

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import ray
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from models import NeuralNetwork
from resource_manager import get_otimized_cpu_workers
from utils import get_cluster_resources


# Training loop.
def train_loop_per_worker(config):

    # Read configurations.
    lr = config["lr"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    # Fetch training dataset.
    train_dataset_shard = ray.train.get_dataset_shard("train")

    # Instantiate and prepare model for training.
    model = config["model"]
    model = ray.train.torch.prepare_model(model)

    # Define loss and optimizer.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create data loader.
    dataloader = train_dataset_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float
    )

    # Train multiple epochs.
    for epoch in range(num_epochs):

        # Train epoch.
        for batch in dataloader:
            output = model(batch["input"])
            loss = loss_fn(output, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create checkpoint.
        base_model = (model.module
            if isinstance(model, DistributedDataParallel) else model)
        checkpoint_dir = tempfile.mkdtemp()
        torch.save(
            {"model_state_dict": base_model.state_dict()},
            os.path.join(checkpoint_dir, "model.pt"),
        )
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

        # Report metrics and checkpoint.
        ray.train.report({"loss": loss.item()}, checkpoint=checkpoint)

if __name__ == "__main__":

    # Define datasets.
    train_dataset = ray.data.from_items(
        [{"input": [x], "label": [2 * x + 1]} for x in range(2000)]
    )
    datasets = {"train": train_dataset}

    # Get Optimized Scaling Parameter
    optimized_workers = get_otimized_cpu_workers()

    # Define configurations.
    train_loop_config = {
        "model": NeuralNetwork(),
        "num_epochs": 20, 
        "lr": 0.01, 
        "batch_size": 32
    }
    scaling_config = ScalingConfig(
        num_workers=optimized_workers, 
        use_gpu=False,
        # placement_strategy="PACK"
    )
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(num_to_keep=1)
    )

    # Initialize the Trainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets=datasets
    )

    # Train the model.
    result = trainer.fit()

    # Inspect the results.
    final_loss = result.metrics["loss"]