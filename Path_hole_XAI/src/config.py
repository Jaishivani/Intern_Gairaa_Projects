from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_FOLDERS = ("normal", "pothole")
CLASS_NAMES = ("NORMAL ROAD", "POTHOLE ROAD")
CLASS_TO_INDEX = {folder: index for index, folder in enumerate(CLASS_FOLDERS)}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DATASET_PATH = PROJECT_ROOT / "dataset"
TRAIN_PATH = DATASET_PATH / "train"
VALIDATION_PATH = DATASET_PATH / "val"
TEST_PATH = DATASET_PATH / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "pothole_resnet18.pth"


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
