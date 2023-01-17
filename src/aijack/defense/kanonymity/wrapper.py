import pandas as pd

from aijack_cpp_core import DataFrame as AnoDataFrame
from aijack_cpp_core import Mondrian as MondrianCore


def convert_pddataframe_to_anodataframe(pd_df, is_continuous_map):
    columns = pd_df.columns.tolist()
    ano_df = AnoDataFrame(columns, is_continuous_map, 0)
    for col in columns:
        if is_continuous_map[col]:
            ano_df.insert_continuous_column(col, pd_df[col].tolist())
        else:
            ano_df.insert_categorical_column(col, pd_df[col].tolist())

    return ano_df


def convert_anodataframe_to_pddataframe(ano_df, feature_columns, is_continuous_map):
    data_continuous = ano_df.get_data_continuous()
    data_categorical = ano_df.get_data_categorical()
    df = pd.DataFrame()
    for col in feature_columns:
        if is_continuous_map[col]:
            df[col] = data_continuous[col]
        else:
            df[col] = data_categorical[col]
    return df


class Mondrian:
    """Implementation of K. LeFevre, D. J. DeWitt and R. Ramakrishnan,
    'Mondrian Multidimensional K-Anonymity,' 22nd International Conference on Data Engineering
    (ICDE'06), Atlanta, GA, USA, 2006, pp. 25-25, doi: 10.1109/ICDE.2006.101. Our implementation
    is based on Nuclearstar/K-Anonymity (https://github.com/Nuclearstar/K-Anonymity)
    """

    def __init__(self, k=3):
        self.api = MondrianCore(k)

    def get_final_partitions(self):
        return self.api.get_final_partitions()

    def anonymize(self, df, feature_columns, sensitive_column, is_continuous_map):
        ano_df = convert_pddataframe_to_anodataframe(df, is_continuous_map)
        ano_anonymized_df = self.api.anonymize(
            ano_df, feature_columns, sensitive_column
        )
        pd_df_anonymized_features = convert_anodataframe_to_pddataframe(
            ano_anonymized_df, feature_columns, is_continuous_map
        )
        pd_df_unused_and_sensitive_columns = df[
            list(set(df.columns) - set(feature_columns))
        ]
        result_df = pd.concat(
            [pd_df_anonymized_features, pd_df_unused_and_sensitive_columns], axis=1
        )
        return result_df
