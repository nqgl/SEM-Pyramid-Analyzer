import grad_like
import numpy as np
import cv2
from t1cv import *
import analysis
import bottleneck as bn
# from sklearn.cluster import (OPTICS, HDBSCAN)
from segmenting_helpers import (get_weighted_mean, calculate_withlen_and_orthlen)



class ImageSegmenter:
    def __init__(self, IA: analysis.ImageAnalysis):
        self.gs: grad_like.ImageGradStats = grad_like.ImageGradStats(IA, 0.08)
        # self.gs.blur_gray()
        self.IA = IA

        
    def est_leg_len_parallel(self, startpoints :np.ndarray, direction_vector:np.ndarray, box_rad = 4, density = 2, steps = 150, iterations = 150, box_spray_velocity = -0.75, velocity_decay = 0.9, doshow=False, threshold_percentile = 50):
        """
        Estimate leg length in a direction for pyramids with centers at startpoints[]. 
        - There are P pyramids
        startpoints: P x 2
        leglens is P x 1
        direction_vector is 2
        startpoints is P x 2
        velocity_scale: P x 2 or 1 x 2
        points will be P*stride x 2
        velocities will be P*stride x 2
        """
        density = 2j * box_rad / density
        P = startpoints.shape[0]
        testpoints = np.mgrid[100 - box_rad : 100 + box_rad: density, 100 - box_rad : 100 + box_rad: density].reshape(2, -1).T
        stride = testpoints.shape[0]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        leglens = np.zeros_like(startpoints[:, 0])
        def make_points_and_velocities(startpoints):
            shape = (P * stride, 2)
            points = np.zeros(shape, dtype = np.float64)
            velocities = np.zeros(shape, dtype = np.float64)
            for i in range(P):
                points[i * stride : (i + 1) * stride, :] = np.mgrid[startpoints[i, 0] - box_rad : startpoints[i, 0] + box_rad: density, startpoints[i, 1] - box_rad : startpoints[i, 1] + box_rad: density].reshape(2, -1).T
            velocities_sample = dirs_to_point(points[0:stride, :], startpoints[0, :]) * box_spray_velocity
            velocities.reshape(P, stride, 2)[:, :, :] = velocities_sample
            return points, velocities

        def velocity_iteration(i, d):
            velocities_to_core = 0 * dirs_to_point(startpoints.reshape(P, 1, 2), d["points"].reshape(P, stride, 2)).reshape(P * stride, 2)
            # velocities_to_core = 1* velocities_to_core / np.sqrt(np.linalg.norm(velocities_to_core, axis = 1).reshape(-1, 1))
            # velocities_to_end = dirs_to_point(point, d["points"])
            # velocities_to_end = 0* velocities_to_end / np.sqrt(np.linalg.norm(velocities_to_core, axis = 1).reshape(-1, 1))
            velocities_toward_line = 0 * dirs_to_line_orth(startpoints.reshape(P, 1, 2), direction_vector, d["points"].reshape(P, stride, 2)).reshape(P * stride, 2)
            # velocities_toward_line = velocities_toward_line / np.sqrt(np.linalg.norm(velocities_toward_line, axis = 1).reshape(-1, 1))
            return 0
            return (velocities_toward_line + velocities_to_core) / steps #* (steps + i) / steps
        new_points = []
        points, velocities = make_points_and_velocities(startpoints)
        landscape = self.gs.sobelxy(ksize=3) / 5
        # landscape += self.gs.sobelxy(ksize=5) / 10
        # landscape += self.gs.npdgrad() *  2
        landscape_cross = np.dstack((landscape[:,:,1], -1 * landscape[:,:,0]))
    
        # landscape += self.gs.npdgrad() * 5
        d = {'points': points, 'velocities': velocities, 'landscape' : landscape, "show":doshow}
    
        steps = 1
        for i in range(steps):
            d = self.gs.hillroll_points(iterations = iterations//steps, velocity_decay = velocity_decay, up = True, **d)
            self.IA.show(d["canvas"], frames = 1)
            d["velocities"] += velocity_iteration(i, d)
        leglens_out = np.zeros_like(leglens, dtype = np.float64)
        orthlens_out = np.zeros_like(leglens, dtype = np.float64)
        
        points = d["points"].reshape(P, stride, 2)
        pointints = d["pointints"].reshape(P, stride, 2)
        active = d["active"].reshape(P, stride)
        for i in range(P):
            l, o = calculate_withlen_and_orthlen(startpoint=startpoints[i], direction_vector=direction_vector,     
                                                      points=points[i], pointints=pointints[i], canvas=d["canvas"], 
                                                      gray=self.IA.gray, active=active[i], new_points=new_points,
                                                      threshold_percentile = threshold_percentile)
            leglens_out[i] = l 
            orthlens_out[i] = o
            if doshow:
                self.IA.show(d["canvas"], ms = 1, frames = 2)

        self.IA.show(d["canvas"], ms = 10, frames = 2)
        return leglens_out, orthlens_out, new_points










import cProfile
# print("running"0)
# cProfile.run('segmenter.segment_with_hill_rolling(1, show =True)')
def segment(IA :analysis.ImageAnalysis, density = 40): 
    gray = IA.gray
    # darken = cv2.imread('darkened_w_circles.png', cv2.IMREAD_GRAYSCALE)

    segmenter = ImageSegmenter(IA)
    # segmenter.skcluster(HDBSCAN(min_cluster_size = 5, cluster_selection_epsilon = 10))
    # segmenter.skcluster(OPTICS(min_samples = 20, max_eps= 300, min_cluster_size=5))
    frames =[]
    IA.capture = frames
    segmenter.gs.gray = gray
    # velocity_scales = [16, 12, -10, 8, -5, 3, 2, -4, 6, -5, -4, 4, -3, 2, -2, 1, 1, -1, -3, -2, -2, 0, 0, 0,0,0,0,0,0,0] + [0] * 25
    velocity_scales = [1] + [-2, 2, -2] + [3] + [-1, 1, -1] * 2 + [0] * 10
    velocity_scales = [1] * 50
    [np.array([0,1]), np.array([1,0]), np.array([0,-1]), np.array([-1,0])]
    # leglen = 1
    #     # leglen, d = segmenter.est_leg_len(np.array([240, 240]), np.array([1, 1]), leglen = leglen, velocity_scale = vs)
    #     leglen, d = segmenter.est_leg_len(np.array([314, 265]), np.array([0, 1]), leglen = leglen, velocity_scale = vs)
    # # leglen = segmenter.est_leg_len(np.array([240, 240]), np.array([1, 1]), leglen = leglen, velocity_scale = vs)
    # segmenter.segment_with_hill_rolling(1, show =True)



    # just wanting a grid of points but arange takes scalars not tuple
    startpoints = np.array(np.meshgrid(np.arange(0, IA.gray.shape[0], density), np.arange(0, IA.gray.shape[1], density))).T.reshape(-1, 2)
    direction_vector = np.array([1, 1])
    # leglen = segmenter.est_leg_len(np.array([240, 240]), np.array([1, 1]), leglen = leglen, velocity_scale = vs)
    MAX_START_POINTS = 10000000
    MIN_BRIGHTNESS_FOR_DUPLICATE_ACCEPT = 175
    MIN_BRIGHTNESS_FOR_TRIPLICATE_ACCEPT = 135
    stability_map = np.zeros_like(IA.gray, dtype = bool)
    skipped_points = []
    stable_points = []
    def add_stable_points(points, reason = "unknown"):
        points = points.reshape(-1, 2)
        for i in range(points.shape[0]):
            p = points[i]
            if not stability_map[p[0], p[1]]:
                stable_points.append(p)
            stability_map[p[0], p[1]] = True
            
    k = -1
    iterations = [75, 75, 75, 500, 75, 100, 200] * 5
    iterations = [50, 50, 200, 300, 200, 500, 600] * 5
    iterations = [100, 150, 100, 150, 100, 100, 300, 200] * 5
    iterations = [1000, 200, 1000, 500] * 4
    iterations = ([1000] + [45, 35, 30] * 15  + [250] * 10 + [500]  + [200] * 2) * 3 + [250] * 10 + [500] * 10 
    iterations = [300, 300, 300, 100, 150, 150, 150, 500, 150] * 5 + [1000, 1500] * 2
    iterations = [30, 30, 300, 50, 200, 200, 325, 250, 500, 400, 200, 140, 180, 50, 70, 60, 70, 50, 20, 15, 20, 30,35] * 4
    iterations = [50, 70, 90, 110, 120, 140, 170 , 300, 300, 500] * 10
    iterations = [70, 30, 60, 40, 30, 30, 300, 250, 30, 76,76] * 10
    iterations = ([65, 56] * 1 + [500, 300, 400, 1000, 100, 150, 600]) * 2
    percentile = 80
    for iters in iterations:
        if len(startpoints) == 0:
            break
        leglens, orthlens, new_points = segmenter.est_leg_len_parallel(startpoints, direction_vector, iterations = iters,
                                                                 velocity_decay=0.96, box_spray_velocity=k, doshow = False,
                                                                 threshold_percentile = percentile)
        # print(leglens)
        k= k * 0.98
        percentile = 1 - (1 - percentile) * 0.99
        orth = np.array([direction_vector[1], -1 * direction_vector[0]])
        orth = orth / np.linalg.norm(orth)
        startpoints = startpoints + (orthlens.reshape(-1, 1) * orth.reshape(1, 2))
        startpoints = startpoints + (leglens.reshape(-1, 1) * direction_vector.reshape(1, 2))
        both_lens = np.dstack((leglens, orthlens)).reshape(-1, 2)
        movement = np.linalg.norm(both_lens, axis = 1)
        stationary = movement < 0.1
        # remove all stationary
        add_stable_points(np.int32(startpoints[stationary, :]), reason = "stationary")
        both_lens = both_lens[~stationary]
        startpoints = startpoints[~stationary, :]
        if len(new_points) > 0 and len(startpoints) < MAX_START_POINTS:
            startpoints = np.concatenate((startpoints, np.array(new_points).reshape(-1, 2)), axis = 0)
            both_lens = np.concatenate((both_lens, np.zeros_like(new_points).reshape(-1,2)), axis = 0)
            # orthlens = np.concatenate((orthlens, np.zeros(len(new_points))), axis = 0)

        else:
            skipped_points += new_points
        # make points into ints for indexing
        startpoints = np.int32(startpoints)
        # remove points that are out of bounds
        inbound_x = np.logical_and(startpoints[:, 0] > 0, startpoints[:, 0] < IA.gray.shape[0])
        inbound_y = np.logical_and(startpoints[:, 1] > 0, startpoints[:, 1] < IA.gray.shape[1])
        inbounds = np.logical_and(inbound_x, inbound_y)
        both_lens = both_lens[inbounds, :]
        startpoints = startpoints[inbounds,:]
        
        # remove points that are already stable
        both_lens = both_lens[~stability_map[startpoints[:, 0], startpoints[:, 1]], :]
        startpoints = startpoints[~stability_map[startpoints[:, 0], startpoints[:, 1]]]
        
        if np.unique(startpoints, axis = 0).shape[0] < startpoints.shape[0]:
            # find all duplicate points:
            unique, uindicies, counts = np.unique(startpoints, axis = 0, return_index=True, return_counts = True)
            duplicates = counts == 2
            brightnesses = gray[unique[:, 0], unique[:, 1]]
            removable_duplicates = np.logical_and(duplicates, brightnesses > MIN_BRIGHTNESS_FOR_DUPLICATE_ACCEPT)
            triplicates_up = counts >= 3
            removable_triplicates = np.logical_and(triplicates_up, brightnesses > MIN_BRIGHTNESS_FOR_TRIPLICATE_ACCEPT)
            removable = np.logical_or(removable_duplicates, removable_triplicates)
            # remove all duplicates

            add_stable_points(unique[removable], reason = "duplicate") # TODO add brigthness check
            both_lens = both_lens[uindicies, :]
            startpoints = startpoints[uindicies, :]
            both_lens = both_lens[~removable, :]
            startpoints = startpoints[~removable, :]
    print(stable_points)
    IA.reset_canvas()
    for p in startpoints:
        IA.draw_cross(p[1], p[0], lengths = [5,5,5,5], color = (250, 0, 0))
    for i in range(len(stable_points)):
        # IA.draw_cross(stable_points[i][0], stable_points[i][1], lengths = [15,15,15,15])   
        IA.draw_cross(stable_points[i][1], stable_points[i][0], lengths = [5,5,5,5], color = (0, 250, 0))
            
    add_stable_points(startpoints, reason = "final")
    IA.render_display(ms = 1000)
    return stable_points

    # import imageio
    # print(len(stable_points))
    # print(len(frames))
    # print()
    # imageio.mimsave("./sample_region_detection.gif", frames, duration = 200)


def main():


    # IA = analysis.ImageAnalysis('437-1-03.tif')  # sparse
    # IA = analysis.ImageAnalysis('244773_02.tif') #dense
    # IA = analysis.ImageAnalysis('244829_02.tif') # mid
    # IA = analysis.ImageAnalysis('darkened.png') # this is the sparse one that gives the issues
    IA = analysis.ImageAnalysis('242316_01.tif') # this is the sparse one that gives the issues
    points = segment(IA)
if __name__ == "__main__":
    main()