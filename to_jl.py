import json

gc = None
with open('gc.json', 'r') as f:
  gc = json.loads(f.read())
  gc = gc['messages']

psf = None
with open('psf.json', 'r') as f:
  psf = json.loads(f.read())
  psf = psf['messages']

tka = None
with open('tka.json', 'r') as f:
  tka = json.loads(f.read())
  tka = tka['messages']

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
