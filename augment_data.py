import pandas as pd
import gender_guesser.detector as gg
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv', low_memory=False)

def add_gender(df):
  detector = gg.Detector(case_sensitive=False)

  gender_map = {
    'male':0,
    'female':1,
    'andy':2,
    'mostly_male':3,
    'mostly_female':4,
    'unknown':5
  }

  genders = []
  for name in df['from']:
    if isinstance(name, str):
      genders.append(
        detector.get_gender(name)
      )
    else:
      genders.append("NO IDEA BUD")

  counter = 0
  gender_dict = {}
  genders_two = []
  for gender in genders:
    if gender not in gender_dict:
      gender_dict[gender] = counter
      counter += 1

    genders_two.append(gender_dict[gender])

  df['gender'] = genders_two

  return df

def clean_text(df):

  text_types = {'plain', 'bold', 'text_link'}
  text = []
  msg_length = []
  for entity in df['text_entities']:
    msg = ""
    for part in entity:
      if part['type'] in text_types:
        msg += f"{part['text']}"

    text.append(msg)

  df['text'] = text

  return df

def add_msg_length(df):
  msg_lengths = []
  for msg in df['text']:
    msg_lengths.append(len(msg))

  df['msg_length'] = msg_lengths
  return df

def add_msg_sentiment(df):
  pass

