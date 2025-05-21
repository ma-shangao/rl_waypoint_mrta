# Copyright 2025 MA Song

import numpy as np
from gps_utils import GPS_utils
from eval import EvalInstance
import pathlib


class RiverTestAlloc:
    def __init__(self):
        self.lon_lat = np.array([
            [-0.3177539, 51.4630214],
            [-0.3182542, 51.4630944],
            [-0.3184723, 51.4629627],
            [-0.3184686, 51.4628235],
            [-0.3183311, 51.4626895],
            [-0.3181908, 51.4626067],
            [-0.3179645, 51.4625671],
            [-0.3178167, 51.4624727],
            [-0.3176474, 51.4625024],
            [-0.3175511, 51.4625945],
            [-0.3177166, 51.4626953],
            [-0.3180112, 51.4628147],
            [-0.3180804, 51.4629295],
            [-0.3180393, 51.4630373],
            [-0.3181010, 51.4631317]
        ])
        self.gu = GPS_utils()
        self.gu.setENUorigin(51.4630214, -0.3177539, 0.0)

    def geo2enuconv(self):
        enu_array = np.zeros((len(self.lon_lat), 2))
        for i in range(len(self.lon_lat)):
            wp = self.lon_lat[i]
            lon, lat = wp
            x_enu, y_enu, z_enu = self.gu.geo2enu(lat, lon, 0.0)
            enu_array[i, 0] = x_enu
            enu_array[i, 1] = y_enu
            print(f"ENU: {x_enu}, {y_enu}, {z_enu}")
            lon, lat, hgt = self.gu.enu2geo(x_enu, y_enu, z_enu)
            print(f"Geo: {lon}, {lat}, {hgt}")
        print(enu_array)
        np.save("enu_array.npy", enu_array)

    def write_waypoints_mav_mission(self):
        # Make sure the dir ./tmp/mav exists
        pathlib.Path('./tmp/mav').mkdir(parents=True, exist_ok=True)

        eval = EvalInstance(problem_data_dir='enu_array.npy')
        tours = eval.eval_single_instance_with_batch_models(3, 'moe_mlp', 128)
        # Convert the tours to geo coordinates
        for i in range(len(tours)):
            f = open(f"./tmp/mav/{i}.txt", "w")
            f.write("QGC WPL 110\n")
            # Fake takeoff point
            f.write("0\t0\t0\t16\t0\t0\t0\t0\t51.4630214\t-0.3177539\t0.0\t1\n")
            tour = tours[i]
            for j in range(len(tour)):
                x_enu, y_enu = tour[j]
                lon, lat, hgt = self.gu.enu2geo(x_enu, y_enu, 0.0)
                print(f"Tour {i}, Waypoint {j}: Geo: {lon}, {lat}, {hgt}")
                f.write(f"{j+1}\t0\t3\t16\t2\t0\t0\t0\t{lon[0]}\t{lat[0]}\t{hgt[0]}\t1\n")


if __name__ == '__main__':
    rt = RiverTestAlloc()
    # rt.geo2enuconv()
    rt.write_waypoints_mav_mission()
