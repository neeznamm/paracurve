import os
import time
import pandas as pd
from pyclick import HumanCurve

OUTPUT_FOLDER = 'synthetic_actions'
MAX_LEN = 128
LEFT = 10
RIGHT = 1600
TOP = 10
BOTTOM = 1200


def generate_trajectories(output_filename):
    f_out = open(output_filename, 'w')
    df = pd.read_csv('statistics/actions_start_stop_1min.csv')
    x_start = df['startx']
    y_start = df['starty']
    x_stop = df['stopx']
    y_stop = df['stopy']
    numpoints = df['length']
    userid = df['userid']
    index = x_start.index
    for i in index:
        from_point = (x_start[i], y_start[i])
        to_point = (x_stop[i], y_stop[i])
        user = userid[i]
        hc = HumanCurve(from_point, to_point)
        points = hc.generateCurve(targetPoints=int(numpoints[i]))

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
        # features.append( str(userid) )
        features = [str(element) for element in features]
        features.append(str(user))
        f_out.write(",".join(features) + '\n')
    f_out.close()


if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_FOLDER)
    except OSError:
        print(OUTPUT_FOLDER + ' folder already exists')
    else:
        print(OUTPUT_FOLDER + ' folder has been created')

    print('Generating humanlike actions')
    output_filename = OUTPUT_FOLDER + '/' + 'bezier_humanlike_actions_seq.csv'
    tic = time.perf_counter()
    generate_trajectories(output_filename)
    toc = time.perf_counter()
    print(f"Execution time: {toc - tic:0.4f} seconds")
