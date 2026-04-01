from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_ADDR = str(REPO_ROOT)
SKYNET_ROOT_ADDR = str(REPO_ROOT)
ADG_SKYNET_ROOT_ADDR = str(REPO_ROOT)
DATA_ROOT = REPO_ROOT / 'data'
RAW_DATA_ROOT = DATA_ROOT / 'raw'
PROCESSED_DATA_ROOT = DATA_ROOT / 'processed'
ARTIFACTS_ROOT = REPO_ROOT / 'artifacts'
