import pandas as pd
import os
from openai import OpenAI

# Load the datasets
covid_misinfo = pd.read_csv('/path/to/covid_misinfo_docs_topic_info.csv')
covid_topic_info = pd.read_csv('/path/to/covid_topic_info.csv')

# Ensure column names are consistent
covid_topic_info.rename(columns={'Topic': 'topic_label'}, inplace=True)

# Select top 5 tweet_texts for each topic_label based on cosine similarity
top_tweets = covid_misinfo.sort_values(by=['topic_label', 'cosine_similarity'], ascending=[True, False])
top_tweets = top_tweets.groupby('topic_label').head(5)

def aggregate_tweets(topic_label):
    tweets = top_tweets[top_tweets['topic_label'] == topic_label]['New_Title'].tolist()
    return '\n'.join([f"{i+1}- {tweet}" for i, tweet in enumerate(tweets)])

# Create a new column that follows the desired structure
covid_topic_info['Updated_Representation'] = covid_topic_info['topic_label'].map(
    lambda x: f"{covid_topic_info[covid_topic_info['topic_label'] == x]['Representation'].values[0]}\n" + aggregate_tweets(x)
)

os.environ['OPENAI_API_KEY'] = 'Put your OPENAI_API_KEY here.'

# Verify it's set by printing it (optional)
print(os.getenv('OPENAI_API_KEY'))
client = OpenAI()

for idx, row in covid_topic_info.iterrows():
      if idx == 0:
        continue
      completion = client.chat.completions.create(
              model="gpt-4o",  # Corrected model name
              messages=[
                  {"role": "system", "content": row['Updated_Representation']},
                  {
                      "role": "user",
                      "content": "Describe topic in a short phrase?"
                  }
              ]
          )

      print(completion.choices[0].message.content)
      covid_topic_info.at[idx, 'topic_description'] = completion.choices[0].message.content

covid_topic_info.to_csv('/path/to/covid_topic_info_description.csv', index=False)