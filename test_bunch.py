import os

input_list = [5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000,
            75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000]

for i in input_list:
    os.system('CUDA_VISIBLE_DEVICES=1 python3 test.py --input-size %s' % str(i))

os.system('CUDA_VISIBLE_DEVICES=1 python3 test.py --real-data True')