from enum import Enum


class ModelState(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    END = "end"


class TrainingState(Enum):
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    TRAIN_EPOCH_START = "on_train_epoch_start"
    TRAIN_EPOCH_END = "on_train_epoch_end"
    VALID_EPOCH_START = "on_valid_epoch_start"
    VALID_EPOCH_END = "on_valid_epoch_end"
    TRAIN_STEP_START = "on_train_step_start"
    TRAIN_STEP_END = "on_train_step_end"
    VALID_STEP_START = "on_valid_step_start"
    VALID_STEP_END = "on_valid_step_end"
    TEST_STEP_START = "on_test_step_start"
    TEST_STEP_END = "on_test_step_end"
