import pandas as pd
import numpy as np
from statistics import mean
import re
import ast
import json
import gender_guesser.detector as gg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from sklearn.preprocessing import LabelEncoder

COMPOUND = 0
NEG = 1
NEU = 2
POS = 3

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
  entities_lists = []
  for entities_list in df['text_entities']:
    jsonified = ast.literal_eval(entities_list)
    entities_lists.append(jsonified)

  for entities_list in entities_lists:
    msg = ""
    for entity in entities_list:
      if entity['type'] in text_types:
        msg += f"{entity['text']}"

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
  compound = []
  neg = []
  neu = []
  pos = []
  for msg in df['text']:
    if not msg:
      continue
    cur_compound = []
    cur_neg = []
    cur_neu = []
    cur_pos = []
    lines_list = tokenize.sent_tokenize(msg)
    print(lines_list)
    for line in lines_list:
      analyzer = SentimentIntensityAnalyzer()
      polarity_scores = analyzer.polarity_scores(line)
      cur_compound.append(polarity_scores['compound'])
      cur_neg.append(polarity_scores['neg'])
      cur_pos.append(polarity_scores['pos'])
      cur_neu.append(polarity_scores['neu'])

    compound.append(mean(cur_compound))
    neg.append(mean(cur_neg))
    neu.append(mean(cur_neu))
    pos.append(mean(cur_pos))

  df['compound'] = compound
  df['neg'] = neg
  df['neu'] = neu
  df['pos'] = pos

  return df
