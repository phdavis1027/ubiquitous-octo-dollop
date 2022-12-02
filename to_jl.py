import json
import datetime as dt
dt = dt.date

gc = None
with open('gc.json', 'r') as f:
  gc = json.loads(f.read())
  gc = gc['messages']

gc_min = min(
  gc,
  key=lambda msg: msg['date_unixtime']
)
print(dt.fromtimestamp(int(gc_min['date_unixtime'])))

psf = None
with open('psf.json', 'r') as f:
  psf = json.loads(f.read())
  psf = psf['messages']

psf_min = min(
  psf,
  key=lambda msg: msg['date_unixtime']
)
print(dt.fromtimestamp(int(psf_min['date_unixtime'])))

tka = None
with open('tka.json', 'r') as f:
  tka = json.loads(f.read())
  tka = tka['messages']

tka_min = min(
  tka,
  key=lambda msg: msg['date_unixtime']
)
print(dt.fromtimestamp(int(tka_min['date_unixtime'])))

samples = 10000
with open('gc.jl', 'w+') as f:
  for message in gc[:samples]:
    message['channel'] = 'gc'
    f.write(json.dumps(message) + '\n')

with open('psf.jl', 'w+') as f:
  for message in psf[:samples]:
    message['channel'] = 'psf'
    f.write(json.dumps(message) + '\n')

with open('tka.jl', 'w+') as f:
  for message in tka[:samples]:
    message['channel'] = 'tka'
    f.write(json.dumps(message) + '\n')
