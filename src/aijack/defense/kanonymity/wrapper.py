import pandas as pd

from aijack_cpp_core import DataFrame as AnoDataFrame
from aijack_cpp_core import Mondrian as MondrianCore


def convert_pddataframe_to_anodataframe(pd_df, is_real_map):
    columns = pd_df.columns.tolist()
    ano_df = AnoDataFrame(columns, is_real_map, 0)
    for col in columns:
        if is_real_map[col]:
            ano_df.insert_real_column(col, pd_df[col].tolist())
        else:
            ano_df.insert_categorical_column(col, pd_df[col].tolist())

    return ano_df


def convert_anodataframe_to_pddataframe(ano_df, is_real_map):
    data_real = ano_df.get_data_real()
    data_categorical = ano_df.get_data_categorical()
    df = pd.DataFrame()
    for col, is_real in is_real_map.items():
        if is_real:
            df[col] = data_real[col]
        else:
            df[col] = data_categorical[col]
    return df


class Mondrian:
    def __init__(self, k=3):
        self.api = MondrianCore(k)

    def anonymize(self, df, feature_columns, sensitive_column, is_real_map):
        ano_df = convert_pddataframe_to_anodataframe(df, is_real_map)
        ano_anonymized_df = self.api.anonymize(
            ano_df, feature_columns, sensitive_column
        )
        pd_anonymized_df = convert_anodataframe_to_pddataframe(
            ano_anonymized_df, is_real_map
        )
        return pd_anonymized_df
