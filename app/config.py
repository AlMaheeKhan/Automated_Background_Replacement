from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_TYPE = "deeplabv3"
MODEL_PATH = BASE_DIR / "models" / "model.pth"

IMAGE_SIZE = (256, 256)

