import pandas as pd

df_test_init = pd.read_csv("./data_casual/test.csv", delimiter=",")
df_train_init = pd.read_csv("./data_casual/train.csv", delimiter=",")
df_test = df_test_init.loc[:, ["input", "label"]]
df_train = df_train_init.loc[:, ["input", "label"]]

input_df_test = df_test_init.loc[:, ["input"]]
output_df_test = df_test_init.loc[:, ["label"]]

input_list_test_before = [x[0] for x in list(input_df_test.values)]
input_list_test = [[
             {"role":"system", "content":"Answer with T if you believe the following statement is correct, answer with F if you believe it is false, do not give any reasoning."},
             {"role":"system", "content":"You are a helpful assistant in casual reasoning, is the following reasoning true?"},
             {"role":"user", "content":f"{x}"}] for x in input_list_test_before]

output_list_test = [str(x[0]) for x in list(output_df_test.values)]

input_df_train = df_train_init.loc[:, ["input"]]
output_df_train = df_train_init.loc[:, ["label"]]

input_list_train_before = [x[0] for x in list(input_df_train.values)]
input_list_train = [[
             {"role":"system", "content":"Answer with T if you believe the following statement is correct, answer with F if you believe it is false, do not give any reasoning."},
             {"role":"system", "content":"You are a helpful assistant in casual reasoning, is the following reasoning true?"},
             {"role":"user", "content":f"{x}"}] for x in input_list_train_before]

output_list_train = [str(x[0]) for x in list(output_df_train.values)]
