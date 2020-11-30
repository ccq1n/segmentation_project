import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, \
    multilabel_confusion_matrix, confusion_matrix, cohen_kappa_score


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        self.val = val
        self.sum += val * ncount
        self.count += ncount
        self.avg = self.sum / self.count


class BinaryClassesCalculation:
    def __init__(self, gt, pred, pred_prob, pos, reverse=False):
        '''

        :param gt: ground truth
        :param pred: pred result
        :param pred_prob: pred prob. result
        :param pos: 0 is health, 1 is pneumonia, 2 is cvoid, set one as positive and other will be negative
        :param reverse: if set as True, pos is neg
        '''

        self.len = len(gt)
        self.gt = np.array(gt)
        self.pred = np.array(pred)
        self.pred_prob = np.array(pred_prob)[:, pos].flatten()

        self.gt[self.gt != pos] = -100
        self.gt[self.gt == pos] = -200
        self.gt[self.gt == -100] = 0
        self.gt[self.gt == -200] = 1

        self.pred[self.pred != pos] = -100
        self.pred[self.pred == pos] = -200
        self.pred[self.pred == -100] = 0
        self.pred[self.pred == -200] = 1

        if reverse:
            self.gt[self.gt == 0] = 2
            self.gt[self.gt == 1] = 0
            self.gt[self.gt == 0] = 1
            self.pred[self.pred == 0] = 2
            self.pred[self.pred == 1] = 0
            self.pred[self.pred == 0] = 1


    def calculate_roc(self):
        try:
            auc = roc_auc_score(self.gt, self.pred_prob)
        except:
            print('error when calculate auc')
            auc = -1.0
        return auc


    def calculate_result(self):
        mat = confusion_matrix(self.gt, self.pred)
        tn, fp, fn, tp = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
        specificity = tn / (fp + tn + 1e-5)
        acc = accuracy_score(self.gt, self.pred)
        f1 = f1_score(self.gt, self.pred)
        precision = precision_score(self.gt, self.pred)
        recall = recall_score(self.gt, self.pred)
        auc = roc_auc_score(self.gt, self.pred_prob)
        kappa = cohen_kappa_score(self.gt, self.pred)
        return acc, f1, precision, recall, specificity, auc, kappa, [tp, tn, tp+fn, fp+tn]


class MetricAIC:
    def __init__(self, y_true, y_pred, y_logit=None, num_class=1):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_logit = np.array(y_logit) if y_logit else None
        self.num_class = num_class


    def calculate_cls_result(self):
        mat = multilabel_confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = mat[:, 0, 0].mean(), mat[:, 0, 1].mean(), mat[:, 1, 0].mean(), mat[:, 1, 1].mean()
        acc = accuracy_score(self.y_true, self.y_pred)
        specificity = tn / (fp + tn + 1e-5)
        f1 = f1_score(self.y_true, self.y_pred, average='macro')
        precision = precision_score(self.y_true, self.y_pred, average='macro')
        recall = recall_score(self.y_true, self.y_pred, average='macro')
        uniques = np.unique(self.y_true)
        auc = 0.0
        if self.y_logit is not None:
            for i in uniques:
                tmp = BinaryClassesCalculation(self.y_true, self.y_pred, self.y_logit, pos=i)
                auc += tmp.calculate_roc()
            auc /= len(uniques)
        kappa = cohen_kappa_score(self.y_true, self.y_pred)
        return acc, f1, precision, recall, specificity, auc, kappa, [tp, tn, tp+fn, fp+tn]

    def calculate_roc(self, pos=2):
        y_true_t = self.y_true.copy()
        y_true_t[self.y_true != pos] = -100
        y_true_t[self.y_true == pos] = -200
        y_true_t[y_true_t == -100] = 0
        y_true_t[y_true_t == -200] = 1

        if self.y_logit:
            y_logit_t = self.y_logit[:, pos].flatten()
            auc = roc_auc_score(y_true_t, y_logit_t)
        else:
            auc = -0.0
        return auc

    def calculate_acc(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc

    def calculate_fpr(self):
        f1 = f1_score(self.y_true, self.y_pred, average='micro')
        precision = precision_score(self.y_true, self.y_pred, average='micro')
        recall = recall_score(self.y_true, self.y_pred, average='micro')
        return f1, precision, recall

    def mask_confusion_matrix(self):
        max_category = self.num_class
        m = np.zeros((max_category, max_category), dtype=int)
        samples = len(self.y_true)
        for i in range(samples):
            c = self.y_pred[i]
            r = self.y_true[i]
            m[r, c] += 1

        return m

    def print_confusion_matrix(self):
        m = self.mask_confusion_matrix()
        num_scans = np.sum(m)
        print('{} scan pairs'.format(num_scans))
        print('')
        print('Prediction')
        for r in range(m.shape[0]):
            s = []
            for c in range(m.shape[1]):
                s.append('{}'.format(m[r, c]))
            print('\t'.join(s))
        print('')
        for i in range(len(m)):
            n = np.diagonal(m, i).sum()
            if i > 0:
                n += np.diagonal(m, -i).sum()
            print('{} categories off: {:.4f}%'.format(i, n / num_scans * 100))
        print('')

        kappa = cohen_kappa_score(self.y_true, self.y_pred, weights='linear')
        print(f'Linear weighted kappa {kappa:.4f}')

    def iou_coef(self):
        epsilon = 1e-6
        y_true_flatten = np.asarray(self.y_true).astype(np.bool)
        y_pred_flatten = np.asarray(self.y_pred).astype(np.bool)

        if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
            return 1.0

        return (np.sum(y_true_flatten * y_pred_flatten) + epsilon) / (
            np.sum(y_true_flatten) + np.sum(y_pred_flatten) - np.sum(y_true_flatten * y_pred_flatten) + epsilon
        )

    def dice_coef(self):
        epsilon = 1e-6
        y_true_flatten = np.asarray(self.y_true).astype(np.bool)
        y_pred_flatten = np.asarray(self.y_pred).astype(np.bool)

        if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
            return 1.0

        return (2. * np.sum(y_true_flatten * y_pred_flatten) + epsilon) / (
            np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon
        )