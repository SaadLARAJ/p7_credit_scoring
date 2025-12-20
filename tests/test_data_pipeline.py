from Src.pipelines.join_datasets import assemble_dataset


def test_joined_dataset_columns():
    df = assemble_dataset()
    expected_columns = {"client_id", "gender", "n_transactions"}
    assert expected_columns.issubset(df.columns)


def test_no_missing_targets():
    df = assemble_dataset()
    assert df["target"].isna().sum() == 0
