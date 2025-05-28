import numpy as np

PURPOSES = ["근무", "정비", "복지", "기타"]
DESTINATIONS = ["본부", "병영", "창고", "식당"]
TIMES = ["주간", "야간"]

PURPOSE_MAP = {p: i for i, p in enumerate(PURPOSES)}
DESTINATION_MAP = {d: i for i, d in enumerate(DESTINATIONS)}
TIME_MAP = {t: i for i, t in enumerate(TIMES)}

INPUT_SIZE = 14
LATENT_SIZE = 4


def encode_info(purpose: str, dest: str, time: str) -> np.ndarray:
    """Encode text info into one-hot numpy array of length 10."""
    p_vec = np.zeros(len(PURPOSES))
    d_vec = np.zeros(len(DESTINATIONS))
    t_vec = np.zeros(len(TIMES))
    p_vec[PURPOSE_MAP[purpose]] = 1
    d_vec[DESTINATION_MAP[dest]] = 1
    t_vec[TIME_MAP[time]] = 1
    return np.concatenate([p_vec, d_vec, t_vec])


