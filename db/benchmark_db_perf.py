"""Performance tests for janky 'database'"""
import json
import glob
import time
import faiss
import numpy as np

parse = [
    "remove the roof on the house",
    "build a floor",
    "build a huge wall around the house",
    "build a huge roof on top of the house"
]

db_size = 10000
n_dim = 768
for i in range(db_size):
    arr = np.random.rand(1, n_dim).astype('float32')
    np.save(f'{i}.npy', arr)


for i in range(db_size):
    with open(f'{i}.json', 'w') as f:
        json.dump({"build a second story on the house": parse}, f)

for i in range(db_size):
    with open(f'{i}.txt', 'w') as f:
        for line in parse:
            f.write(f'{line}\n')

# Performance test with parses stored as json files

start_json_load = time.time()
index = faiss.IndexFlatL2(n_dim)
for fname in glob.glob('*.npy'):
    index.add(np.load(fname))
seqs = {}
for fname in glob.glob('*.json'):
    with open(fname, 'r') as f:
        seq = json.load(f)
    seqs.update(seq)
end_json_load = time.time()

json_load_time = end_json_load - start_json_load
print(f"Time to load npy and json into FAISS: {json_load_time}")

# Performance test with parses stored as txt files

start_txt_load = time.time()
index = faiss.IndexFlatL2(n_dim)
for fname in glob.glob('*.npy'):
    index.add(np.load(fname))
seqs = []
for fname in glob.glob('*.txt'):
    with open(fname, 'r') as f:
        seq = f.readlines()
        seq = [s.strip() for s in seq]
        seqs.append(seq)
end_txt_load = time.time()

txt_load_time = end_txt_load - start_txt_load
print(f"Time to load npy and txt into FAISS: {txt_load_time}")



