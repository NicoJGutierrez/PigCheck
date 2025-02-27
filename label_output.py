import pandas as pd
import re

prompt_output = pd.read_json("answers.json")

# Add flags
prompt_output['good_environment'] = prompt_output['environment'].str.contains('outdoor|straw', case=False, na=False).astype(int)
prompt_output['welfare_red_flag'] = prompt_output.apply(lambda row: 1 if len(row['health_issues']) > 0 or len(row['negative_behaviours']) > 0 else 0, axis=1)

# Sort welfare issues on top
prompt_output.sort_values('welfare_red_flag', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
prompt_output = prompt_output.reset_index(drop = True)

print(prompt_output.good_environment.value_counts())
print(prompt_output.welfare_red_flag.value_counts())

prompt_output.to_csv("labelled_output.csv", index = False)