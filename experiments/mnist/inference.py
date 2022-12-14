import numpy as np
from torch.utils.data import DataLoader

from experiment import load_test_dataset, MNISTInference
from nasge import utils as nasge_utils

if __name__ == "__main__":
    logger = nasge_utils.get_logger("Inference")

    config = nasge_utils.load_yaml("./experiments/mnist/config.yaml")
    parameters = config["parameters"]

    test_dataset = load_test_dataset("./data",)
    test_dloader = DataLoader(test_dataset,
                              batch_size=parameters["test_batch_size"],
                              shuffle=True)
    inferencer = MNISTInference(parameters["model_path"], test_dloader)
    lbls, preds = inferencer.inference()

    accuracy = np.sum(preds == lbls) / preds.shape[0]
    logger.info(f"accuracy: {accuracy}")
    logger.info(inferencer.model)
