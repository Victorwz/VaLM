import random
import os

f_train = open('split_data/train.txt', 'w')
f_valid = open('split_data/valid.txt', 'w')
f_test = open('split_data/test.txt', 'w')


def assign_text(text):
    point = random.random()
    if point < 0.00005:
        f_valid.write(text)
    elif point < 0.00025:
        f_test.write(text)
    else:
        f_train.write(text)


for root,dirs,files in os.walk('raw_data/'):
    for file in files:
        print(f'preprocessing file {os.path.join(root, file)}.....')
        with open(os.path.join(root, file), 'r') as f:
            acc_line = ''
            line_cnt = 0
            while True:
                line = f.readline()
                if not line:
                    break
                line_cnt += 1
                acc_line += line
                if line_cnt % 1000 == 0:
                    assign_text(acc_line)
                    acc_line = ''
            if acc_line != '':
                print(f'============== {line_cnt} append trim last ==============')
                assign_text(acc_line)
                acc_line = ''

f_train.close()
f_valid.close()
f_test.close()