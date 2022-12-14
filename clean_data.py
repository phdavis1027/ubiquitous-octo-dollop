import re

from to_jl import *
from csv_maker import *

def clean_text(df):
    return df.apply(remove_entities, axis=1)

def remove_entities(row):
    text = row['text']
    if text.__str__() == "nan":
        return row

    entities = row['text_entities'][1:-1].split('},')
    for i, e in enumerate(entities):
        if e and not e.endswith('}'):
            entities[i] += "}"
        
        # print(type(text))
        # print(text)
        match = re.search(r".*'plain', 'text': '(.*|\n*|\\*|\**)'[},]", text)
        # print(match)
        if match:
            rtext = match.group(1)
            # print(rtext)
            text = re.sub(e, rtext, text)

    
    if row['text'] != text:
        print(text)
        print(row['text'])
    row['text'] = text
    return row
