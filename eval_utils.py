from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))


def sat_evaluation(pred, label, sat_num):
    acc = sum([int(p == l) for p, l in zip(pred, label)]) / len(label)
    precision = precision_score(label, pred, average='macro', zero_division=0)
    sk_recall = recall_score(label, pred, average='macro', zero_division=0)
    f1 = f1_score(label, pred, average='macro', zero_division=0)
    #     sat_result = (acc, precision, sk_recall, f1)

    recall = [[0, 0] for _ in range(sat_num)]
    for p, l in zip(pred, label):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]

    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(pred, label)
    spearman_val = spearman(pred, label)

    bi_pred = [int(item < sat_num // 2) for item in pred]
    bi_label = [int(item < sat_num // 2) for item in label]
    bi_recall = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if l == 1]) / max(bi_label.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if p == 1]) / max(bi_pred.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

    sat_result = [UAR, kappa, spearman_val, bi_f1, acc, precision, sk_recall, f1]
    return sat_result
