def test_mondrian():
    import pandas as pd

    from aijack.defense.kanonymity import Mondrian

    # This test code is based on https://github.com/glassonion1/anonypy

    data = [
        [6, "1", "test1", "x", 20],
        [6, "1", "test1", "x", 30],
        [8, "2", "test2", "x", 50],
        [8, "2", "test3", "w", 45],
        [8, "1", "test2", "y", 35],
        [4, "2", "test3", "y", 20],
        [4, "1", "test3", "y", 20],
        [2, "1", "test3", "z", 22],
        [2, "2", "test3", "y", 32],
    ]

    columns = ["col1", "col2", "col3", "col4", "col5"]
    feature_columns = ["col1", "col2", "col3"]
    is_continuous_map = {
        "col1": True,
        "col2": False,
        "col3": False,
        "col4": False,
        "col5": True,
    }
    sensitive_column = "col4"

    df = pd.DataFrame(data=data, columns=columns)

    mondrian = Mondrian(k=2)
    adf_ignore_unused_features = mondrian.anonymize(
        df, feature_columns, sensitive_column, is_continuous_map, True
    )
    adf_not_ignore_unused_features = mondrian.anonymize(
        df, feature_columns, sensitive_column, is_continuous_map, False
    )

    test_adf_data_ignore_unused_features = pd.DataFrame(
        [
            [3.0, "1", "test3", "z"],
            [3.0, "1", "test3", "y"],
            [3.0, "2", "test3", "y"],
            [3.0, "2", "test3", "y"],
            [6.666666507720947, "1", "test1_test2", "x"],
            [6.666666507720947, "1", "test1_test2", "x"],
            [6.666666507720947, "1", "test1_test2", "y"],
            [8.0, "2", "test2_test3", "x"],
            [8.0, "2", "test2_test3", "w"],
        ],
        columns=["col1", "col2", "col3", "col4"],
    )

    assert adf_ignore_unused_features.equals(test_adf_data_ignore_unused_features)

    test_adf_data_not_ignore_unused_features = pd.DataFrame(
        [
            [3.0, "1", "test3", "z", 20],
            [3.0, "1", "test3", "y", 30],
            [3.0, "2", "test3", "y", 50],
            [3.0, "2", "test3", "y", 45],
            [6.666666507720947, "1", "test1_test2", "x", 35],
            [6.666666507720947, "1", "test1_test2", "x", 20],
            [6.666666507720947, "1", "test1_test2", "y", 20],
            [8.0, "2", "test2_test3", "x", 22],
            [8.0, "2", "test2_test3", "w", 32],
        ],
        columns=["col1", "col2", "col3", "col4", "col5"],
        index=[7, 6, 8, 5, 0, 1, 4, 2, 3],
    )

    assert adf_not_ignore_unused_features.equals(
        test_adf_data_not_ignore_unused_features
    )
