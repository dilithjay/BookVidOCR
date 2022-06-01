import os

frame_dir = "./masked_data/"
file_data = "filename, page\n"
for filename in sorted(os.listdir(frame_dir)):
    if filename.endswith('.jpg'):
        file_data += f'{filename}, \n'

with open("ground_truth/gt1.csv", 'w') as f:
    f.write(file_data)
