from quietblink.splits import SIM_CALIBRATION_SEEDS, SIM_TEST_SEEDS, SUBJECTS, split_subjects


def test_subject_split_is_disjoint_complete_and_stable():
    calibration, test = split_subjects()
    assert not set(calibration) & set(test)
    assert sorted(calibration + test) == list(SUBJECTS)
    assert (calibration, test) == split_subjects()


def test_simulator_seed_splits_are_disjoint():
    assert not set(SIM_CALIBRATION_SEEDS) & set(SIM_TEST_SEEDS)
