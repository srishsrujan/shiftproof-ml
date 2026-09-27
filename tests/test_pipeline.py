import pandas as pd

from data.generate_data import generate_requests
from src.shiftproof.data_pipeline import build_preprocessor, prepare_xy, time_split


def test_time_split_is_chronological():
    df = generate_requests(n=300)
    splits = time_split(df)
    assert splits.train.created_at.max() <= splits.validation.created_at.min()
    assert splits.validation.created_at.max() <= splits.test.created_at.min()


def test_preprocessor_handles_missing_and_unknown_categories():
    df = generate_requests(n=400)
    splits = time_split(df)
    x_train, _ = prepare_xy(splits.train)
    x_test, _ = prepare_xy(splits.test)
    pre = build_preprocessor()
    xt = pre.fit_transform(x_train)
    x2 = x_test.copy()
    x2["request_type"] = "brand_new_category"
    out = pre.transform(x2)
    assert xt.shape[1] == out.shape[1]
