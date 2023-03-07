import os
import time
import pandas as pd
import concurrent.futures
from pyclick import HumanCurve

OUTPUT_FOLDER = 'synthetic_actions'
MAX_LEN = 128
LEFT = 10
RIGHT = 1600
TOP = 10
BOTTOM = 1200


def init(output_filename):
    df = pd.read_csv('statistics/actions_start_stop_1min.csv')

    chunk_size = 650
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        with open(output_filename, 'w') as f_out:
            for future in concurrent.futures.as_completed(futures):
                features_list = future.result()
                for features in features_list:
                    features_str = [str(f) for f in features]
                    f_out.write(",".join(features_str) + '\n')
        executor.shutdown()
    f_out.close()


def process_chunk(chunk):
    x_start = chunk['startx']
    y_start = chunk['starty']
    x_stop = chunk['stopx']
    y_stop = chunk['stopy']
    numpoints = chunk['length']
    userid = chunk['userid']
    index = x_start.index

    features_list = []
    for i in index:
        from_point = (x_start[i], y_start[i])
        to_point = (x_stop[i], y_stop[i])
        user = userid[i]

        hc = HumanCurve(from_point, to_point)
        points = hc.generateCurve(targetPoints=int(numpoints[i]))

        # Process the points_x and points_y arrays
        points_x = [int(pair[0]) for pair in points]
        points_y = [int(pair[1]) for pair in points]
        dx = pd.Series(points_x).diff()
        dy = pd.Series(points_y).diff()
        first_row = dx.index[0]
        dx = dx.drop(first_row)
        dy = dy.drop(first_row)
        # Series --> list
        dx = dx.values.tolist()
        dy = dy.values.tolist()
        if len(dx) < MAX_LEN:
            # shorter are padded with 0s
            for j in range(MAX_LEN - len(dx)):
                dx.append(0)
                dy.append(0)
        else:
            # longer are shortened to MAX_LEN
            dx = dx[0:MAX_LEN]
            dy = dy[0:MAX_LEN]
        features = []
        features.extend(dx)
        features.extend(dy)
        features.append(str(user))
        features_list.append(features)
    return features_list


if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_FOLDER)
    except OSError:
        print(OUTPUT_FOLDER + ' folder already exists')
    else:
        print(OUTPUT_FOLDER + ' folder has been created')

    print('Generating humanlike actions')
    output_filename = OUTPUT_FOLDER + '/' + 'bezier_humanlike_actions_para.csv'
    tic = time.perf_counter()
    init(output_filename)
    toc = time.perf_counter()
    print(f"Execution time: {toc - tic:0.4f} seconds")
