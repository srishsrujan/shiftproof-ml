import pandas as pd

from data.generate_data import generate_students
from src.shiftproof.data_pipeline import build_preprocessor, prepare_xy, time_split


def test_time_split_is_chronological():
    df = generate_students(n=300)
    splits = time_split(df)
    assert splits.train.graduation_year.max() <= splits.validation.graduation_year.min()
    assert splits.validation.graduation_year.max() <= splits.test.graduation_year.min()


def test_preprocessor_handles_missing_and_unknown_categories():
    df = generate_students(n=400)
    splits = time_split(df)
    x_train, _ = prepare_xy(splits.train)
    x_test, _ = prepare_xy(splits.test)
    pre = build_preprocessor()
    xt = pre.fit_transform(x_train)
    x2 = x_test.copy()
    x2["branch"] = "brand_new_category"
    out = pre.transform(x2)
    assert xt.shape[1] == out.shape[1]
