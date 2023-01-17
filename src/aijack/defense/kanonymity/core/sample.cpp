#include "anonymizer.h"

int main()
{

    int num_row, num_col;
    std::cin >> num_row >> num_col;

    std::vector<std::string> columns(num_col);
    std::map<std::string, bool> is_continuous;
    bool tmp_continuous_flag;
    for (int j = 0; j < num_col; j++)
    {
        std::cin >> columns[j];
        std::cin >> tmp_continuous_flag;
        is_continuous.insert(std::make_pair(columns[j], tmp_continuous_flag));
    }

    DataFrame df = DataFrame(columns, is_continuous);

    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
        {
            if (is_continuous[columns[j]])
            {
                float tmp_value;
                std::cin >> tmp_value;
                df.insert_continuous(columns[j], tmp_value);
            }
            else
            {
                std::string tmp_value;
                std::cin >> tmp_value;
                df.insert_categorical(columns[j], tmp_value);
            }
        }
    }

    df.print();
    std::cout << "---" << std::endl;

    vector<string> tmp_col = {"age", "disease"};
    vector<int> tmp_idx_1 = {0, 2};
    std::pair<std::vector<std::string>, std::vector<int>> tmp_indices =
        std::make_pair(tmp_col, tmp_idx_1);
    df[tmp_indices].print();
    std::cout << "---" << std::endl;

    std::vector<int> tmp_idx = {0, 1, 2, 3, 4};
    std::pair<vector<int>, vector<int>> tmp_split_result_1 =
        split_dataframe(df, tmp_idx, "age");
    for (int i = 0; i < tmp_split_result_1.first.size(); i++)
    {
        std::cout << tmp_split_result_1.first[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < tmp_split_result_1.second.size(); i++)
    {
        std::cout << tmp_split_result_1.second[i] << " ";
    }
    std::cout << std::endl;

    std::vector<string> feature_columns = {"age", "sex", "zip"};
    Mondrian mondrian = Mondrian(3);
    DataFrame anonymized_df = mondrian.anonymize(df, feature_columns, "disease");
    anonymized_df.print();
}
