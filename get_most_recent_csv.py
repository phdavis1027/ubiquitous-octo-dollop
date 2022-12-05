import os
import datetime

most_recent = datetime.datetime(1980,9,1,0,0,0)
sample_string ="XXXX-XX-XX"
for results in os.listdir('result'):
  result_date = results[:len(sample_string) + 1]
  result_time = results[len(sample_string) + 1:]
  candidate = f"{result_date} {result_time}"
  print(candidate)
  candidate = datetime.datetime.strptime(
    f"{result_date} {result_time}",
    "%y-%m-%d %H:%M:%s"
  )
