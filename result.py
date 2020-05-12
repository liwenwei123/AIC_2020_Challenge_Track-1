import pandas as pd
import os

for video_id in range(1,32):
    csv_file_name = os.path.join('output', '{}.csv'.format(video_id))
    data = pd.read_csv(csv_file_name, encoding='utf-8')
    data.sort_values(by = 'frame_id',axis = 0,ascending = True, inplace = True)
    with open('track1.txt','a+', encoding='utf-8') as f:
        for line in data.values:
            if line[3] != line[3]:
                line[3] = 1
            f.write((" ".join(str(int(i)) for i in line))+'\n')
