# coding: utf-8

CPU_CORES = 2

EPISODE_LENGTH = 30
FEATURE_NUM = 5
DATA_BUFFER_SIZE = 20000

SIM_DATA_DIR = './sim_data'
MODEL_DATA_DIR = './model_data'

SIM_ROUNDS = 1000  # total sample size: SIM_ROUNDS * EPISODE_LENGTH
SIM_BATCH_SIZE = CPU_CORES
SIM_ROUNDS_PER_STEP = 23

IMPROVE_STEPS_PER_EPOCH = 100
IMPROVE_BATCH_SIZE = 2048

VALID_ROUNDS = 100
VALID_ROUNDS_PER_STEP = 23
VALID_INTERVAL = 600

CURRENT_MODEL_FILE = 'model.current'
