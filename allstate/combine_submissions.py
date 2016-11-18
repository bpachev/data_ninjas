from sys import argv, exit
import pandas as pd

ids = None
tot = len(argv)-2

if tot <= 1:
    print "Must combine more than one submission"

loss = None
for filename in argv[1:-1]:
  df = pd.read_csv(filename)
  print filename, df
  if loss is None:
      loss = df["loss"].as_matrix()
  else:
      loss += df["loss"].as_matrix()
  ids = df["id"]

loss /= float(tot)
print loss
pd.DataFrame({"id":ids, "loss":loss}).to_csv(argv[-1], index= False)
