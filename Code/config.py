from dataclasses import dataclass

@dataclass
class Config:
    RANDOM_SEED: int = 123

    TRAIN_DATA_PATH: str = '../../train_data/train_data'
    GT_PATH: str = '../../train_data/train_gt.csv'
    WL_PATH: str = '../../train_data/wavelengths.csv'
    SUBMISSION_DIR: str = '../Submissions'
    TARGET_COLS: list = ['P', 'K', 'Mg', 'pH']

    SIGMA_PARAM = (0, 2, 2) 
    SG_WINDOW = 11
    SG_POLY = 2
    SG_DERIV = 1


