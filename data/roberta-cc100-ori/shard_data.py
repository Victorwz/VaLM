import random
import os

f40 = [None for i in range(40)]

for i in range (40):
    f40[i] = open(f'shard/train.{i}.txt', 'w')


def random_fileno():
    return random.randint(0, 39)


fileno = 0
with open('raw/train.txt') as f:
    line_cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        f40[fileno].write(line)
        line_cnt += 1
        if line_cnt % 1000 == 0:
            fileno = random_fileno()

for i in range (40):
    f40[i].close()