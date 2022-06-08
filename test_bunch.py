import os

c_in = 64
c_out = 128

input_list = [5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000,
            75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 400000, 500000]

for i in input_list:
    os.system('python3 test.py --input-size %s --in-channels %s --out-channels %s' % (str(i), str(c_in), str(c_out)))

os.system('python3 test.py --real-data True --in-channels %s --out-channels %s' % (str(c_in), str(c_out)))