

def load_dblp_labels():
    f1 = '../dataset/DBLP/data/author_label.txt'
    f2 = '../dataset/DBLP/data/paper_label.txt'
    f3 = '../dataset/DBLP/data/conf_label.txt'

    labels = {}
    for line in open(f1).readlines():
        line = line.strip().split('\t')
        node = str('A' + line[0])
        labels[node] = line[1]

    for line in open(f2).readlines():
        line = line.strip().split('\t')
        node = str('P' + line[0])
        labels[node] = line[1]

    for line in open(f3).readlines():
        line = line.strip().split('\t')
        node = str('C' + line[0])
        labels[node] = line[1]

    return labels