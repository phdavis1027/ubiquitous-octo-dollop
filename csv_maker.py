import pandas as pd
import ciso8601 as ciso
import time as time

start_date = '20220525T000000'

tka_df = pd.read_json('tka.jl', lines=True)
tka_df = tka_df.sample(frac=1, axis=1).reset_index(drop=True)
tka_df = tka_df[
    tka_df['date_unixtime'] > time.mktime(ciso.parse_datetime(start_date).timetuple())
]

gc_df = pd.read_json('gc.jl', lines=True)
gc_df = gc_df.sample(frac=1, axis=1).reset_index(drop=True)
gc_df = gc_df[
    gc_df['date_unixtime'] > time.mktime(ciso.parse_datetime(start_date).timetuple())
]

psf_df = pd.read_json('psf.jl', lines=True)
psf_df = psf_df.sample(frac=1, axis=1).reset_index(drop=True)
psf_df = psf_df[
    psf_df['date_unixtime'] > time.mktime(ciso.parse_datetime(start_date).timetuple())
]

df = pd.concat(
    [tka_df, gc_df, psf_df],
    axis=0
)



df.to_csv('data.csv')
