import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import AverageMeter, intersection_and_union, intersection_and_union_gpu, make_dirs

from sklearn.cluster import MeanShift
import torch.distributed as dist
import pickle
TESTERS = Registry("testers")

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def merge_evaluator(evaluator, tmp_dir, prefix=''):
    rank, world_size = get_dist_info()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()
    pickle.dump(evaluator, open(os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    for i in range(1, world_size):
        part_file = os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, i))
        evaluator.merge(pickle.load(open(part_file, 'rb')))

    return evaluator


def meanshift_cluster(shifted_pcd, valid, bandwidth=1.2):
    embedding_dim = shifted_pcd.shape[1]
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, embedding_dim) if valid is not None else shifted_pcd
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        ms.fit(valid_shifts)
    except Exception as e:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(valid_shifts)
        print("\nException: {}.".format(e))
        print("Disable bin_seeding.")
    labels = ms.labels_ + 1
    assert np.min(labels) > 0
    if valid is not None:
        clustered_ins_ids[valid] = labels
        return clustered_ins_ids
    else:
        return labels

def printResults(class_evaluator, logger=None, sem_only=False, condition="nuScenes"):
    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = class_evaluator.getPQ()
    class_IoU, class_all_IoU = class_evaluator.getSemIoU()

    # now make a nice dictionary
    output_dict = {}
    if condition == "nuScenes":
        things = ['barrier','bicycle', 'bus',  'car', 'construction_vehicle','motorcycle', 'pedestrian','traffic_cone','trailer','truck']
        stuff = ['driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
        class_strings  =  {  0: 'noise',
                            1: 'barrier',
                            2: 'bicycle',
                            3: 'bus',
                            4: 'car',
                            5: 'construction_vehicle',
                            6: 'motorcycle',
                            7: 'pedestrian',
                            8: 'traffic_cone',
                            9: 'trailer',
                            10: 'truck',
                            11: 'driveable_surface',
                            12: 'other_flat',
                            13: 'sidewalk',
                            14: 'terrain',
                            15: 'manmade',
                            16: 'vegetation'}
    else:
        things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
        stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole','traffic-sign'
                ]

        class_strings  =  { 
                            0:"ignore", 
                            1:"car", 
                            2:"bicycle",
                            3: "motorcycle",
                            4: "truck",
                            5: "other-vehicle",
                            6: "person",
                            7:  "bicyclist", 
                            8:  "motorcyclist", 
                            9:  "road", 
                            10:  "parking",
                            11: "sidewalk",
                            12:  "other-ground", 
                            13: "building", 
                            14: "fence",
                            15: "vegetation",
                            16:  "trunk", 
                            17: "terrain",
                            18: "pole",
                            19:"traffic-sign"}
    all_classes = things + stuff
    # make python variables
    class_PQ = class_PQ.item()
    class_SQ = class_SQ.item()
    class_RQ = class_RQ.item()
    class_all_PQ = class_all_PQ.flatten().tolist()
    class_all_SQ = class_all_SQ.flatten().tolist()
    class_all_RQ = class_all_RQ.flatten().tolist()
    class_IoU = class_IoU.item()
    class_all_IoU = class_all_IoU.flatten().tolist()

    output_dict["all"] = {}
    output_dict["all"]["PQ"] = class_PQ
    output_dict["all"]["SQ"] = class_SQ
    output_dict["all"]["RQ"] = class_RQ
    output_dict["all"]["IoU"] = class_IoU

    classwise_tables = {}

    for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
        class_str = class_strings[idx]
        output_dict[class_str] = {}
        output_dict[class_str]["PQ"] = pq
        output_dict[class_str]["SQ"] = sq
        output_dict[class_str]["RQ"] = rq
        output_dict[class_str]["IoU"] = iou

    PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
    PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
    RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
    SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])

    PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
    RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
    SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])

    PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
    RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
    SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])
    mIoU = output_dict["all"]["IoU"]

    codalab_output = {}
    codalab_output["pq_mean"] = float(PQ_all)
    codalab_output["pq_dagger"] = float(PQ_dagger)
    codalab_output["sq_mean"] = float(SQ_all)
    codalab_output["rq_mean"] = float(RQ_all)
    codalab_output["iou_mean"] = float(mIoU)
    codalab_output["pq_stuff"] = float(PQ_stuff)
    codalab_output["rq_stuff"] = float(RQ_stuff)
    codalab_output["sq_stuff"] = float(SQ_stuff)
    codalab_output["pq_things"] = float(PQ_things)
    codalab_output["rq_things"] = float(RQ_things)
    codalab_output["sq_things"] = float(SQ_things)

    key_list = [
        "pq_mean",
        "pq_dagger",
        "sq_mean",
        "rq_mean",
        "iou_mean",
        "pq_stuff",
        "rq_stuff",
        "sq_stuff",
        "pq_things",
        "rq_things",
        "sq_things"
    ]

    if sem_only and logger != None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        logger.info('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |  IoU   |   PQ   |   RQ   |   SQ   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU'], v['PQ'], v['RQ'], v['SQ']
            ))
        return codalab_output
    if sem_only and logger is None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        print('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        print('|        |  IoU   |   PQ   |   RQ   |   SQ   |')
        for k, v in output_dict.items():
            print('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU'], v['PQ'], v['RQ'], v['SQ']
            ))
        return codalab_output

    if logger != None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        logger.info('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |   PQ   |   RQ   |   SQ   |  IoU   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['PQ'], v['RQ'], v['SQ'], v['IoU']
            ))
        logger.info('True Positive: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_tp]))
        logger.info('False Positive: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_fp]))
        logger.info('False Negative: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_fn]))
    if logger is None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        print('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        print('|        |   PQ   |   RQ   |   SQ   |  IoU   |')
        for k, v in output_dict.items():
            print('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['PQ'], v['RQ'], v['SQ'], v['IoU']
            ))
        print('True Positive: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_tp]))
        print('False Positive: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_fp]))
        print('False Negative: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_fn]))

    for key in key_list:
        if logger != None:
            logger.info("{}:\t{}".format(key, codalab_output[key]))
        else:
            print("{}:\t{}".format(key, codalab_output[key]))

    return codalab_output


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)


class PanopticEval:
    """ Panoptic evaluation using numpy

    authors: Andres Milioto and Jens Behley

    """

    def __init__(self, n_classes, device=None, ignore=None, offset=2 ** 32, min_points=30):
        self.n_classes = n_classes
        assert (device == None)
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)


        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = min_points  # smallest number of points to consider instances in gt
        self.eps = 1e-15

        self.original_results = []

    def num_classes(self):
        return self.n_classes

    def merge(self, evaluator):
        self.px_iou_conf_matrix += evaluator.px_iou_conf_matrix
        self.pan_tp += evaluator.pan_tp
        self.pan_iou += evaluator.pan_iou
        self.pan_fp += evaluator.pan_fp
        self.pan_fn += evaluator.pan_fn

        self.evaluated_fnames += evaluator.evaluated_fnames

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        self.evaluated_fnames = []

    ################################# IoU STUFF ##################################
    def addBatchSemIoU(self, x_sem, y_sem):
        # import  pdb;pdb.set_trace()
        # idxs are labels and predictions
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes, since the predictions
        # go on the rows)
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        # print(f"tp={tp}")
        # print(f"fp={fp}")
        # print(f"fn={fn}")
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()

        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getSemAcc(self):
        tp, fp, fn = self.getSemIoUStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum()
        total = np.maximum(total, self.eps)
        acc_mean = total_tp.astype(np.double) / total.astype(np.double)

        return acc_mean  # returns "acc mean"

    ################################# IoU STUFF ##################################
    ##############################################################################

    #############################  Panoptic STUFF ################################
    def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore:
            # make a mask for this class
            gt_not_in_excl_mask = y_sem_row != cl
            # remove all other points
            x_sem_row = x_sem_row[gt_not_in_excl_mask]
            y_sem_row = y_sem_row[gt_not_in_excl_mask]
            x_inst_row = x_inst_row[gt_not_in_excl_mask]
            y_inst_row = y_inst_row[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in self.include:
            # print("*"*80)
            # print("CLASS", cl.item())
            # get a class mask
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float64) / unions.astype(np.float64)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

            # count the FP)
            self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

    def getPQ(self):
        # first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
            self.eps)
        pq_all = sq_all * rq_all

        # then do the REAL mean (no ignored classes)
        SQ = sq_all[self.include].mean()
        RQ = rq_all[self.include].mean()
        PQ = pq_all[self.include].mean()

        return PQ, SQ, RQ, pq_all, sq_all, rq_all

    #############################  Panoptic STUFF ################################
    ##############################################################################

    def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
        ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
        '''
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)

    def addBatch_w_fname(self, x_sem, x_inst, y_sem, y_inst, fname):  # x=preds, y=targets
        ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
        '''
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)

        self.evaluated_fnames.append(fname)


def init_eval(min_points=20, condition="nuScenes"):
    if condition == "nuScenes":
        nr_classes = 17
    else:
        nr_classes = 20 
    ignore_class = [0]
    # print("New evaluator with min_points of {}".format(min_points))
    class_evaluator = PanopticEval(nr_classes, None, ignore_class, min_points=min_points)
    return class_evaluator


def eval_one_scan_w_fname(class_evaluator, gt_sem, gt_ins, pred_sem, pred_ins):
    class_evaluator.addBatch(pred_sem, pred_ins, gt_sem, gt_ins)

def update_evaluator(evaluator, sem_preds, ins_preds, inputs, segment):
    for i in range(len(sem_preds)):
        # import  pdb;pdb.set_trace()
        eval_one_scan_w_fname(evaluator, segment.reshape(-1),
                            inputs['inst_data'].reshape(-1).numpy(),
                            sem_preds[i].reshape(-1), ins_preds[i].reshape(-1))

class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.logger = get_root_logger(log_file=os.path.join(cfg.save_path, "test.log"),
                                      file_mode='a' if cfg.resume else 'w')
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(model.cuda(),
                                 broadcast_buffers=False,
                                 find_unused_parameters=self.cfg.find_unused_parameters)
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info("=> Loaded weight '{}' (epoch {})".format(self.cfg.weight, checkpoint['epoch']))
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self.cfg.batch_size_test_per_gpu,
                                                  shuffle=False,
                                                  num_workers=self.cfg.batch_size_test_per_gpu,
                                                  pin_memory=True,
                                                  sampler=test_sampler,
                                                  collate_fn=self.__class__.collate_fn)
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)

@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        
        condition = "SemanticKITTI"
        # panoptic
        # if nuscnes
        if self.cfg.data.test.type == "NuScenesDataset":
            condition = "nuScenes"
        if condition == "nuScenes":
            min_points = 20
            valid_xentropy_ids = [0,1, 4, 2, 3, 5, 6, 7, 8, 9]
            bandwidth = 2.5
        
        # # kitti
        else:
            valid_xentropy_ids = [0, 1, 4, 2, 3, 5, 6, 7]
            min_points = 50
            bandwidth = 1.2
        before_merge_evaluator = init_eval(min_points=min_points, condition=condition)


        output_dir = './output'
        tmp_dir = os.path.join(output_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)


        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if self.cfg.data.test.type == 'ScanNetDataset' and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json
            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(meta=dict(
                use_camera=False, use_lidar=True, use_radar=False, use_map=False, use_external=False
            ))
            with open(os.path.join(save_path, "submit", "test", "submission.json"), "w") as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, '{}_pred.npy'.format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(self.test_loader), data_name))
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, len(fragment_list))
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs: be], :] += pred_part[bs: be]
                        bs = be
                    logger.info('Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}'.format(
                        idx + 1, len(self.test_loader), data_name=data_name, batch_idx=i, batch_num=len(fragment_list)))
                pred = pred.max(1)[1].data.cpu().numpy()
                # np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                segment = data_dict["origin_segment"]
            
            panoptic_eval = False
            if "pano_dict" in data_dict:
                pano_dict = data_dict["pano_dict"]
                panoptic_eval = True
            if panoptic_eval:
                # print("start to evaluate panoptic >>>>>>>>>>>>>>>>>>>>>>>")
                pano_dict["condition"] = [pano_dict["condition"]]
                pano_dict["discrete_coord"] = pano_dict["discrete_coord"].cuda()
                pano_dict["grid_coord"] = pano_dict["grid_coord"].cuda()
                pano_dict["feat"] = pano_dict["feat"].cuda()
                # pano_dict["gt_off"] = pano_dict["gt_off"].cuda()
                pano_dict["offset"] = pano_dict["offset"].cuda()
                pano_dict["inverse_indexes"] = pano_dict["inverse_indexes"].cuda()

                ind = 0
                batch_index = []
                batch_index.append(ind)
                batch_index.append(pano_dict["org_coord"].shape[0])
                pano_dict["batch_index"] = batch_index

                pred_offsets = self.model(pano_dict)["pred_offsets"]
                pt_cart_xyz = pano_dict['org_coord']
                pt_pred_offsets = [pred_offsets[i].detach().cpu().numpy().reshape(-1, 3) for i in
                                range(len(pred_offsets))]
                
                pt_pred_valid = []
                for i in range(len(pred_offsets)):
                        pt_pred_valid.append(np.isin(pred, valid_xentropy_ids).reshape(-1))
                pred_ins_ids_list = []
                for i in range(len(pred_offsets)):
                    i_clustered_ins_ids = meanshift_cluster(pt_cart_xyz.numpy() + pt_pred_offsets[i], pt_pred_valid[i],
                                                            bandwidth=bandwidth)
                    pred_ins_ids_list.append(i_clustered_ins_ids)
                update_evaluator(before_merge_evaluator, [pred+1], pred_ins_ids_list, pano_dict, segment+1)


            intersection, union, target = intersection_and_union(pred, segment, self.cfg.data.num_classes,
                                                                 self.cfg.data.ignore_index)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}]-{} '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {acc:.4f} ({m_acc:.4f}) '
                        'mIoU {iou:.4f} ({m_iou:.4f})'.format(data_name, idx + 1, len(self.test_loader), segment.size,
                                                              batch_time=batch_time, acc=acc, m_acc=m_acc,
                                                              iou=iou, m_iou=m_iou))
            if self.cfg.data.test.type == "ScanNetDataset":
                np.savetxt(os.path.join(save_path, "submit", '{}.txt'.format(data_name)),
                           self.test_loader.dataset.class2id[pred].reshape([-1, 1]), fmt="%d")
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                # os.makedirs(
                #     os.path.join(save_path, "submit", "sequences", sequence_name, "predictions"), exist_ok=True
                # )
                # pred = pred.astype(np.uint32)
                # pred = np.vectorize(self.test_loader.dataset.learning_map_inv.__getitem__)(pred).astype(np.uint32)
            #     pred.tofile(
            #         os.path.join(save_path, "submit", "sequences", sequence_name, "predictions", f"{frame_name}.label")
            #     )
            # elif self.cfg.data.test.type == "NuScenesDataset":
            #     np.array(pred + 1).astype(np.uint8).tofile(
            #         os.path.join(save_path, "submit", "lidarseg", "test", '{}_lidarseg.bin'.format(data_name)))

        logger.info("Syncing ...")
        comm.synchronize()
        intersection_meter_sync = comm.gather(intersection_meter, dst=0)
        union_meter_sync = comm.gather(union_meter, dst=0)
        target_meter_sync = comm.gather(target_meter, dst=0)
        if panoptic_eval:
            before_merge_evaluator = merge_evaluator(before_merge_evaluator, tmp_dir)
     
        if comm.is_main_process():
            if panoptic_eval:
                before_merge_results = printResults(before_merge_evaluator, logger=logger, sem_only=True,condition=condition)
                logger.info(before_merge_results)


            intersection = np.sum([meter.sum for meter in intersection_meter_sync], axis=0)
            union = np.sum([meter.sum for meter in union_meter_sync], axis=0)
            target = np.sum([meter.sum for meter in target_meter_sync], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                    idx=i, name=self.cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(pred, label, self.cfg.data.num_classes,
                                                                     self.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info('Test: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f} '.format(i + 1, len(self.test_loader),
                                                          batch_time=batch_time,
                                                          accuracy=accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

        for i in range(self.cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=self.cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch))
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min((i + 1) * self.cfg.batch_size_test, len(data_dict_list))
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(
                    data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}] '
                        'Batch {batch_time.val:.3f} '
                        '({batch_time.avg:.3f}) '.format(
                data_name, idx + 1, len(self.test_loader), batch_time=batch_time))

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info('Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.'.format(ins_mIoU, cat_mIoU))
        for i in range(num_categories):
            logger.info('Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}'.format(
                idx=i, name=self.test_loader.dataset.categories[i],
                iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                iou_count=int(iou_count[i])))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
