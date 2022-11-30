import pandas as pd

tka_df = pd.read_json('tka.jl', lines=True)
gc_df = pd.read_json('gc.jl', lines=True)
psf_df = pd.read_json('psf.jl', lines=True)

df = pd.concat(
    [tka_df, gc_df, psf_df],
    axis=0
)

df.to_csv('data.csv')
