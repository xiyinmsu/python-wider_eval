# Written by Xi Yin @ Microsoft
# Apr. 29, 2019
import os
import os.path as op
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import time
import pdb


def load_gt_mat_to_lists(gt_dir):
    gt_mat = loadmat(op.join(gt_dir, 'wider_face_val.mat'))
    easy_mat = loadmat(op.join(gt_dir, 'wider_easy_val.mat'))
    medium_mat = loadmat(op.join(gt_dir, 'wider_medium_val.mat'))
    hard_mat = loadmat(op.join(gt_dir, 'wider_hard_val.mat'))
    event_list = [_[0][0] for _ in gt_mat['event_list']]
    file_list = []
    facebox_list = []
    easy_list = []
    medium_list = []
    hard_list = []
    for file_list_per_event, box_list_per_event, easy_list_per_event, \
            median_list_per_event, hard_list_per_event in zip(
            gt_mat['file_list'], gt_mat['face_bbx_list'],
            easy_mat['gt_list'], medium_mat['gt_list'], hard_mat['gt_list']
        ):
        file_list.append([_[0][0] for _ in file_list_per_event[0]])
        facebox_list.append([_[0] for _ in box_list_per_event[0]])
        easy_list.append([_[0].tolist() if not _[0].tolist() else
                          np.concatenate(_[0]).tolist()
                          for _ in easy_list_per_event[0]])
        medium_list.append([_[0].tolist() if not _[0].tolist() else
                          np.concatenate(_[0]).tolist()
                          for _ in median_list_per_event[0]])
        hard_list.append([_[0].tolist() if not _[0].tolist() else
                          np.concatenate(_[0]).tolist()
                          for _ in hard_list_per_event[0]])
    set_gt_lists = [easy_list, medium_list, hard_list]
    return event_list, file_list, facebox_list, set_gt_lists


def read_pred(pred_dir, event_list, file_list, score_thresh=0.0):
    # support to specify a minimum threshold to select detection
    # results with confidence larger than score_thresh for 
    # evaluation. 
    event_num = len(event_list)
    pred_list = []
    for i in range(event_num):
        print("Read prediction: current event %d"%i)
        img_list = file_list[i]
        img_num = len(img_list)
        box_list = []
        for j in range(img_num):
            pred_file = op.join(pred_dir, event_list[i], img_list[j]+'.txt')
            if not op.isfile(pred_file):
                print("Cannot find the prediction file {}".format(pred_file))
                continue
            with open(pred_file, 'r') as f:
                lines = f.readlines()
            try:
                bbx_num = int(lines[1].strip())
                bbx = []
                for k in range(bbx_num):
                    raw_info = lines[k+2].strip().split(',')
                    if float(raw_info[-1]) >= score_thresh:
                        bbx.append([float(_) for _ in raw_info])
                # sort the box in each image in desending order of confidence
                bbx = sorted(bbx, key=lambda x:-x[-1])
                box_list.append(np.array(bbx))
            except:
                box_list.append([])
                print("Invalid format of prediction file {}".format(pred_file))
        pred_list.append(box_list)
    return pred_list


def norm_score(org_pred_list):
    # get min and max of scores
    all_scores = [box[-1] for event in org_pred_list for img in event for box in img]
    min_score = min(all_scores)
    max_score = max(all_scores)
    event_num = len(org_pred_list)
    norm_pred_list = []
    for i in range(event_num):
        print("Normalize prediction scores: current event %d"%(i))
        pred_list = org_pred_list[i]
        for j, img_list in enumerate(pred_list):
            if len(img_list) == 0:
                continue
            # min max normalization to [0,1]
            img_list[:,4] = (img_list[:,4] - min_score) / (max_score - min_score)
        norm_pred_list.append(pred_list)
    return norm_pred_list


def boxoverlap(boxlist, box):
    x1 = np.maximum(boxlist[:,0], box[0])
    y1 = np.maximum(boxlist[:,1], box[1])
    x2 = np.minimum(boxlist[:,2], box[2])
    y2 = np.minimum(boxlist[:,3], box[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    overlap = np.zeros(boxlist.shape[0])
    valid = (w >= 0) * (h >= 0)
    inter = w[valid] * h[valid]
    aarea = (boxlist[valid,2] - boxlist[valid,0] + 1) * (boxlist[valid,3] - boxlist[valid,1] + 1)
    barea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    overlap[valid] = inter / (aarea + barea - inter)
    return overlap


def image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh):
    pred_num = pred_info.shape[0]
    gt_num = gt_bbx.shape[0]
    # change box mode from xywh to xyxy
    pred_info[:,2] = pred_info[:,0] + pred_info[:,2]
    pred_info[:,3] = pred_info[:,1] + pred_info[:,3]
    gt_bbx[:,2] = gt_bbx[:,0] + gt_bbx[:,2]
    gt_bbx[:,3] = gt_bbx[:,1] + gt_bbx[:,3]
    pred_recall = np.zeros(pred_num)
    recall_list = np.zeros(gt_num)
    proposal_list = np.ones(pred_num)
    cnt = 0
    for h in range(pred_num):
        overlap_list = boxoverlap(gt_bbx, pred_info[h][:4])
        idx = np.argmax(overlap_list)
        if overlap_list[idx] >= IoU_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
                cnt += 1
        pred_recall[h] = cnt
    return pred_recall, proposal_list


def image_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    img_pr_info = np.zeros([thresh_num, 2])
    thresholds = np.linspace(1-1.0/thresh_num, 0.0, thresh_num)
    pred_num = pred_info.shape[0]
    num = 0
    for t in range(thresh_num):
        thresh = thresholds[t]
        indexes = np.where(pred_info[:,4] >= thresh)[0]
        if indexes.size > num:
            r_index = np.max(indexes)
            p_index_sum = sum(proposal_list[:r_index+1]==1)
            img_pr_info[t,0] = p_index_sum
            img_pr_info[t,1] = pred_recall[r_index]
            num = indexes.size
        elif num > 0:
            # skip the above logic if r_index does not change.
            # this helps to speed up evaluation. 
            img_pr_info[t,0] = img_pr_info[t-1, 0]
            img_pr_info[t,1] = img_pr_info[t-1, 1]
    return img_pr_info


def dataset_pr_info(thresh_num, org_pr_curve, count_face):
    pr_curve = np.zeros([thresh_num, 2])
    pr_curve[:,0] = org_pr_curve[:,1] / org_pr_curve[:,0]
    pr_curve[:,1] = org_pr_curve[:,1] / count_face
    return pr_curve


def calc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap


def evaluation(norm_pred_list, facebox_list, set_gt_list,
               set_name, setting_class, method, settings):
    IoU_thresh = settings['IoU_thresh']
    thresh_num = settings['thresh_num']
    event_num = len(facebox_list)
    count_face = 0
    org_pr_curve = np.zeros([thresh_num, 2])
    for i in range(event_num):
        print("Evaluating %s set at event %d"%(set_name, i))
        gt_bbx_list = facebox_list[i]
        pred_list = norm_pred_list[i]
        sub_gt_list = set_gt_list[i]
        img_num = len(gt_bbx_list)
        img_pr_info_list = []
        for j in range(img_num):
            gt_bbx = gt_bbx_list[j].copy()
            pred_info = pred_list[j].copy()
            gt_bbx = np.reshape(gt_bbx, (-1, 4))
            pred_info = np.reshape(pred_info, (-1, 5))
            keep_index = sub_gt_list[j]
            count_face += len(keep_index)
            if gt_bbx.shape[0]==0 or pred_info.shape[0]==0:
                continue
            # matlab index to python index
            keep_index_py = [_ - 1 for _ in keep_index]
            ignore = [1 if _ in keep_index_py else 0 for _ in range(len(gt_bbx))]
            pred_recall, proposal_list = image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh)
            img_pr_info = image_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
            img_pr_info_list.append(img_pr_info)

            if len(img_pr_info) != 0:
                org_pr_curve += img_pr_info

    pr_curve = dataset_pr_info(thresh_num, org_pr_curve, count_face)   
    # save pr_curve results for plot.
    # use the official scripts for figure plotting.
    # note the typo in MATLAB scripts (pr_cruve).
    res = {
            'legend_name': method,
            'pr_cruve': pr_curve
          }
    method_path = op.join('eval_tools/plot/baselines/Val/', setting_class, method)
    if not op.isdir(method_path):
        os.mkdir(method_path)

    save_file = op.join(method_path, 'wider_pr_info_{}_{}.mat'.format(method, set_name))
    savemat(save_file, res)

    return pr_curve


def wider_eval(gt_dir, pred_dir, method, settings, score_thresh=0.0, save_file=None):
    setting_name_list = settings['setting_name_list']
    setting_class = settings['setting_class']
    dataset_class = settings['dataset_class']

    # load gt mat files to list representations
    event_list, file_list, facebox_list, set_gt_lists = load_gt_mat_to_lists(gt_dir)
    # load prediction text file
    pred_list = read_pred(pred_dir, event_list, file_list, score_thresh)
    # score normalization
    norm_pred_list = norm_score(pred_list)

    setting_aps = []
    for i, set_name in enumerate(setting_name_list):
        print("Current evaluation setting {}".format(set_name))
        set_gt_list = set_gt_lists[i]
        pr_curve = evaluation(norm_pred_list, facebox_list, set_gt_list,
                   set_name, setting_class, method, settings)
        ap = calc_ap(pr_curve[:,1], pr_curve[:,0])
        setting_aps.append(ap)
    
    # save results to txt for future reference
    if save_file is not None:
        if op.isdir(save_file):
            save_file = op.join(save_file, 'result.txt')
    else:
        save_file = 'result.txt'

    with open(save_file, 'w') as f:
        f.write("AP\n")
        f.write("Easy: {}\n".format(setting_aps[0]))
        f.write("Medium: {}\n".format(setting_aps[1]))
        f.write("Hard: {}\n".format(setting_aps[2]))

    print("==================== AP Results ===================")
    print("{} on {}: Easy   AP = {}".format(method, dataset_class, setting_aps[0]))
    print("{} on {}: Medium AP = {}".format(method, dataset_class, setting_aps[1]))
    print("{} on {}: Hard   AP = {}".format(method, dataset_class, setting_aps[2]))
    print("===================================================")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='WiderFace evaluation.')
    parser.add_argument('-g', '--gt_dir', required=False, type=str,
                        default='eval_tools/ground_truth/',
                        help='ground truth dir for mat files')
    parser.add_argument('-p', '--pred_dir', required=False, type=str,
                        default='eval_tools/pred/',
                        help='prediction file dir')
    parser.add_argument('-m', '--method_name', required=False, type=str,
                        default='Ours',
                        help='method name, default=Ours')
    parser.add_argument('-s', '--score_thresh', required=False, type=float,
                        default=0.0, 
                        help='min threshold to select detection results')
    parser.add_argument('-f', '--save_file', required=False, type=str,
                        default=None, 
                        help='filename to save final mAP')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    This evaluation script follows the exact logic from the official 
    evaluation tools in order to reproduce the same results.  
    There are minor changes in several places to speed up evaluation.
    """
    start = time.time()
    args = parse_args()
    settings = {
        'setting_name_list': ['easy_val', 'medium_val', 'hard_val'],
        'setting_class': 'setting_int',
        'dataset_class': 'Val',
        'IoU_thresh': 0.5,
        'thresh_num': 1000
        }
    wider_eval(args.gt_dir, args.pred_dir, args.method_name, settings, 
            args.score_thresh, args.save_file)
    end = time.time()
    print("Elapsed time: {}".format(end - start))

