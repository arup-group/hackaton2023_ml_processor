import numpy as np
import csv

from smt.sampling_methods import LHS

# Define bounds for the arrays
xlimits = np.array([[2.0, 10.0], [5.0, 12.0], [5.0, 12.0], [0.1, 0.4]])
sampling = LHS(xlimits=xlimits)

# Define the number of samples
num = 100
x = sampling(num)

# Export to csv
with open('lhs_sampling_3.csv', 'w') as file:

    writer = csv.writer(file)
    for i in x:
        writer.writerow(i)