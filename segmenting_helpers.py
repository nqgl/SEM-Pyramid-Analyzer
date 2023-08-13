import numpy as np
import cv2
from t1cv import *
import bottleneck as bn

rng = np.random.default_rng()


def get_weighted_mean(unique_points, unique_point_count, magnitudes, gray):
    brightnesses = np.float32(gray[unique_points[:, 0], unique_points[:, 1]].reshape(-1))
    min_brightness, max_brightness = bn.nanmin(brightnesses), bn.nanmax(brightnesses)
    if max_brightness == min_brightness:
        if magnitudes.shape[0] == 0:
            return 0
        else:
            return np.mean(magnitudes)
    brightness_rank = (brightnesses - min_brightness + 1) / (max_brightness - min_brightness)
    weights = brightness_rank * unique_point_count
    mean = np.average(magnitudes, axis=0, weights=weights)
    return mean

def get_weighted_median(unique_points, unique_point_count, magnitudes, gray):
    brightnesses = np.float32(gray[unique_points[:, 0], unique_points[:, 1]].reshape(-1))
    min_brightness, max_brightness = bn.nanmin(brightnesses), bn.nanmax(brightnesses)
    brightness_rank = (brightnesses - min_brightness + 1) / (max_brightness - min_brightness)
    weights = brightness_rank * unique_point_count
    magnitude_args = np.argsort(magnitudes)
    target = sum(weights)
    cumsum = np.cumsum(weights[magnitude_args])
    median_index = np.searchsorted(cumsum, target / 2)
    median = magnitudes[magnitude_args[median_index]]
    return median

def calculate_withlen_and_orthlen(startpoint, direction_vector, points, pointints, canvas, gray, active, new_points, threshold_percentile = 50):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    pointints_active = pointints[active]
    unique_points, unique_point_indicies, unique_point_count = np.unique(pointints_active, axis=0, return_index=True, return_counts=True)
    brightness_levels = gray[pointints_active[:, 0], pointints_active[:, 1]]
    try:
        # unique_bright = gray[unique_points[:, 0], unique_points[:, 1]]
        # brightness_threshold = bn.nanmedian(unique_bright)
        # brightness_threshold = max(brightness_threshold, bn.nanmedian(brightness_levels),bn.nanmean(brightness_levels)) - 5
        brightness_threshold = np.percentile(brightness_levels, threshold_percentile, interpolation='lower')

    except:
        print(np.sum(brightness_levels))
        input("brightness_threshold error")
        brightness_threshold = 0
    # brightness_threshold = 0
    bright_enough = (brightness_levels >= brightness_threshold).reshape(-1)
    bright_enough_unique = (brightness_levels[unique_point_indicies] >= brightness_threshold).reshape(-1)
    # print(bright_enough.shape, orth_dists.shape)
    # orth_dist_quartiles = np.percentile(orth_dists, [25, 75])
    kept_pointints = pointints_active[bright_enough, :]
    canvas[kept_pointints[:, 0], kept_pointints[:, 1], :] = np.uint8((0, 0, 255))
    kept_half = bright_enough
    # print(kept_half.shape)
    # kept_pointints_unique = unique_points[bright_enough]
    # canvas[kept_pointints[:, 0], kept_pointints[:, 1], :] = np.uint8((0, 255, 0))
    orth_dists = scalars_to_line_orth(startpoint, direction_vector, pointints_active).reshape(-1)
    orth_dists_unique = orth_dists[unique_point_indicies[bright_enough_unique]]
    magnitudes = scalars_to_with_line(startpoint, direction_vector, pointints_active).reshape(-1)
    magnitudes_unique = magnitudes[unique_point_indicies[bright_enough_unique]]
    # magnitudes = magnitudes[kept_half[:]]
    if len(magnitudes) == 0:
        return 0, 0

    medweight = 4
    meanweight = 9
    withlen_median = get_weighted_median(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], magnitudes_unique, gray)
    withlen_mean = get_weighted_mean(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], magnitudes_unique, gray)
    orthlen_median = get_weighted_median(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], orth_dists_unique, gray)
    orthlen_mean = get_weighted_mean(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], orth_dists_unique, gray)
    DISTDIFF_MIN = 8
    split_leg, split_orth = None, None
    orth = np.array([direction_vector[1], -direction_vector[0]])
    withlen = (withlen_mean * meanweight + withlen_median * medweight) / (medweight + meanweight)
    orthlen = (orthlen_mean * meanweight + orthlen_median * medweight) / (medweight + meanweight)
    # if abs(withlen_median - withlen_mean) > DISTDIFF_MIN and not (0.5 < withlen_median / withlen_mean < 2):
    #     split_leg = withlen_median
    #     withlen = withlen_mean
    # if abs(orthlen_median - orthlen_mean) > DISTDIFF_MIN and not (0.5 < orthlen_median / orthlen_mean < 2):
    #     split_orth = orthlen_median
    #     if split_leg is None:
    #         new_points.append(startpoint + orthlen_mean * orth + withlen * direction_vector)
    #     else:
    #         new_points.append(startpoint + orthlen_mean * orth + withlen * direction_vector)
    #         new_points.append(startpoint + orthlen_mean * orth + split_leg * direction_vector)
    #         new_points.append(startpoint + orthlen_median * orth + split_leg * direction_vector)
    medpoint = np.array((withlen_median, orthlen_median))
    meanpoint = np.array((withlen_mean, orthlen_mean))
    thispoint = np.array((withlen, orthlen))
    # if np.linalg.norm(thispoint) < 2.5 and np.linalg.norm(medpoint - meanpoint) > 3:
    if np.linalg.norm(thispoint) * 2 < np.linalg.norm(medpoint - meanpoint) > 1:
        if rng.random() < 0.5:
            withlen = withlen_mean * (1 + rng.random())
        else:
            withlen = withlen_median * (1 + rng.random())
        if rng.random() < 0.5:
            orthlen = orthlen_mean * (1 + rng.random())
        else:
            orthlen = orthlen_median * (1 + rng.random())
        # new_points.append(startpoint + orthlen_median * orth + withlen_median * direction_vector)
    
    #     orthlen = orthlen_mean
    if withlen == np.inf or withlen == -np.inf or withlen == np.nan or withlen == float("nan") or len(magnitudes) == 0:
        print("withlen error", withlen, withlen_mean, withlen_median, len(magnitudes))
        # input()
        return 0, 0
    arrow(canvas, startpoint, startpoint + withlen_mean * direction_vector, (200, 80, 0), 3)
    arrow(canvas, startpoint, startpoint + withlen_median * direction_vector, (25, 20, 150), 2)
    arrow(canvas, startpoint + withlen_mean * direction_vector, startpoint + orthlen_mean * orth + withlen_mean * direction_vector, (200, 80, 0), 3)
    arrow(canvas, startpoint + withlen_median * direction_vector, startpoint + orthlen_median * orth + withlen_median * direction_vector, (25, 20, 150), 2)

    
    return withlen, orthlen


