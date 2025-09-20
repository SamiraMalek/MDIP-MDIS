import pandas as pd
import os
from openai import OpenAI

# Load the datasets
hpv_topic_info = pd.read_csv('/path/to/covid_topic_info_description.csv')

# Filter out rows where topic_label is -1
hpv_topic_info = hpv_topic_info[hpv_topic_info['topic_label'] != -1]

# Convert dataframe to desired format
output = "\n".join(hpv_topic_info.apply(lambda row: f"{row['topic_label']}- {row['topic_description']}", axis=1))

os.environ['OPENAI_API_KEY'] = 'put your OPENAI_API_KEY here'

# Verify it's set by printing it (optional)
print(os.getenv('OPENAI_API_KEY'))
client = OpenAI()

completion = client.chat.completions.create(
              model="gpt-4o",  # Corrected model name
              messages=[
                  {"role": "system", "content": "The following are topics related to COVID19. Go through all topics and categorize them into relevant groups. Mention topics number for each category."},
                  {
                      "role": "user",
                      "content": output
                  }
              ]
          )

print(completion.choices[0].message.content)