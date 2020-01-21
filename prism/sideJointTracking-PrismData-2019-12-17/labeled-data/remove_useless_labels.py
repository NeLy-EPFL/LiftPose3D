import os

import cv2
import csv
import numpy as np

if __name__ == '__main__':
    dirs = [d for d in os.listdir() if (os.path.isdir(d)) and (d[0] != '.')]
    dirs.sort()

    all_images = []
    all_labels = []

    for d in dirs:
        print("working in %s"%d)
        
        # first left, then right
        labels = []
        with open(d+"/CollectedData_PrismData.csv", 'r') as csv_labels,\
            open(d+"/CollectedData_PrismData_new.csv", 'w') as csv_labels_new:
            csv_reader = csv.reader(csv_labels, delimiter=',')
            csv_writer = csv.writer(csv_labels_new, delimiter=',')
            i = 0
            for row in csv_reader:
                if i >= 3:
                    row_new = []
                    row_np = np.array(list(map(lambda r: '0' if r == '' else r, row[1:]))).astype(float)
                    row_np[row_np != 0] = 1
                    left = np.sum(row_np[:30])
                    right = np.sum(row_np[30:])
                    
                    if left == 0 and right == 0:
                        csv_writer.writerow(row)
                        continue

                    row_new.append(row[0])
                    if left > right:
                        row_new += row[1:31]
                        row_new += ['']*30
                    else:
                        row_new += ['']*30
                        row_new += row[31:]
                    csv_writer.writerow(row_new)
                else:
                    csv_writer.writerow(row)

                i += 1
