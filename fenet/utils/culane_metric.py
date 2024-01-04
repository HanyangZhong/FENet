import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img,
                 tuple(p1),
                 tuple(p2),
                 color=(255, 255, 255),
                 thickness=width)
    return img

# whole view
def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


import numpy as np
import cv2  # Ensure OpenCV is imported

def discrete_cross_iou_new(xs, ys, width=30, img_shape=(590, 1640, 3), far_weight=0.6, near_weight=0.4,partial_view=1):
    # Calculate the cut-off point for 'far' and 'near' (3/4 of the image height)
    if partial_view == 1:
        # 1/2 partial view
        cut_off = img_shape[0] * 2 // 3
    elif partial_view == 2:
        # 1/3 partial view
        cut_off = img_shape[0] * 3 // 4
    elif partial_view == 3:
        # 1/4 partial view
        cut_off = img_shape[0] * 5 // 8


    # Check if there are any predicted lanes and actual lanes
    if not (xs.size>0 and ys.size>0):
        # If there are no predictions or no actual lanes, return zeros for the IoU scores
        return np.zeros((max(len(xs), len(ys)), max(len(xs), len(ys))))

    # Define a function to extract 'far' and 'near' points based on the cut-off
    def extract_sections(lane, cut_off):
        far_part = [point for point in lane if point[1] < cut_off]
        near_part = [point for point in lane if point[1] >= cut_off]
        return np.array(far_part, dtype=np.int32), np.array(near_part, dtype=np.int32)  # Convert to NumPy array

    # Separate the 'far' and 'near' parts of the lanes
    xs_far, xs_near = zip(*[extract_sections(lane, cut_off) for lane in xs])
    ys_far, ys_near = zip(*[extract_sections(lane, cut_off) for lane in ys])

    # Initialize IoU matrices for 'far' and 'near' with the same shape
    max_lanes = max(len(xs), len(ys))
    ious_far = np.zeros((max_lanes, max_lanes))
    ious_near = np.zeros((max_lanes, max_lanes))

    # Calculate IoU for each pair of lanes for 'far' and 'near', only if there are points to draw
    for i, x_far in enumerate(xs_far):
        if len(x_far) > 1:
            x_far_mask = draw_lane(x_far, img_shape=img_shape, width=width) > 0
            for j, y_far in enumerate(ys_far):
                if len(y_far) > 1:
                    y_far_mask = draw_lane(y_far, img_shape=img_shape, width=width) > 0
                    ious_far[i, j] = (x_far_mask & y_far_mask).sum() / (x_far_mask | y_far_mask).sum()

    for i, x_near in enumerate(xs_near):
        if len(x_near) > 1:
            x_near_mask = draw_lane(x_near, img_shape=img_shape, width=width) > 0
            for j, y_near in enumerate(ys_near):
                if len(y_near) > 1:
                    y_near_mask = draw_lane(y_near, img_shape=img_shape, width=width) > 0
                    ious_near[i, j] = (x_near_mask & y_near_mask).sum() / (x_near_mask | y_near_mask).sum()

    # Combine the 'far' and 'near' IoUs with the specified weights
    combined_ious = far_weight * ious_far[:len(xs), :len(ys)] + near_weight * ious_near[:len(xs), :len(ys)]

    return ious_far[:len(xs), :len(ys)]
    # return ious_near[:len(xs), :len(ys)]


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3),partial_view_num=0):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    # old
    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    # print('old pred:',interp_pred.shape)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (4, 50, 2)
    # print('old anno:',interp_anno.shape)

    if official:
        if partial_view_num == 0:
            ious = discrete_cross_iou(interp_pred,
                                      interp_anno,
                                      width=width,
                                      img_shape=img_shape)
        else:
            ious = discrete_cross_iou_new(interp_pred,
                                    interp_anno,
                                    width=width,
                                    img_shape=img_shape,partial_view=partial_view_num)
    else:
        ious = continuous_cross_iou(interp_pred,
                                    interp_anno,
                                    width=width,
                                    img_shape=img_shape)
    # print('ious is:',ious)
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric

def load_culane_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data

def load_culane_img_data_special(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]
    # print(img_data)

    return img_data

def load_culane_data(data_dir, file_list_path):
    # 加载所有路径
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(
                data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace(
                    '.jpg', '.lines.txt')) for line in file_list.readlines()
        ]

    data = []
    for path in filepaths:
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data

def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False,
                     partial_view_num=0):
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Calculating metric for List: {}'.format(list_path))
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    img_shape = (590, 1640, 3)
    if sequential:
        results = map(
            partial(culane_metric,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,
                    img_shape=img_shape,partial_view_num=partial_view_num), predictions, annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric, zip(predictions, annotations,
                        repeat(width),
                        repeat(iou_thresholds),
                        repeat(official),
                        repeat(img_shape)),partial_view_num=partial_view_num)

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        # for m in results:
        #     print('m is',m)
        #     a = round(m*100)/100
        #     print(a)
        #     tp = sum(a[thr][0])
        fp = sum(m[thr][1] for m in results)

        fn = sum(m[thr][2] for m in results)

        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(total_tp, total_fp,
            total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential)

        header = '=' * 20 + ' Results ({})'.format(
            os.path.basename(list_path)) + '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True)
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True)
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        nargs='+',
                        help="Path to txt file containing the list of files",
                        required=True)
    parser.add_argument("--sequential",
                        action='store_true',
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        action='store_true',
                        help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == '__main__':
    main()
