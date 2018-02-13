import json, ast
import pickle

with open('qMap_inceptionV1', 'r') as hfile:
  qmap = json.load(hfile)
  print(qmap)
  strMap = {}
  for k in qmap:
    strMap[str(k)] = str(qmap[k])
  
  print(strMap)



