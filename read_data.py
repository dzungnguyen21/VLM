import json 

with open("llava_instruct_150k.json") as f:
    data = json.load(f)
    print(data[0])


    