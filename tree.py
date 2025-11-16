from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

# ---------- 1. 基础函数（熵、划分、选最优特征、建树） ----------
def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        label = feat_vec[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    shannon_ent = 0.0
    for count in label_counts.values():
        prob = count / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced = feat_vec[:axis] + feat_vec[axis+1:]
            ret_dataset.append(reduced)
    return ret_dataset

def choose_best_feature(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_gain, best_feature = 0.0, -1
    for i in range(num_features):
        feat_vals = [vec[i] for vec in dataset]
        unique_vals = set(feat_vals)
        new_entropy = 0.0
        for val in unique_vals:
            sub_set = split_dataset(dataset, i, val)
            prob = len(sub_set) / len(dataset)
            new_entropy += prob * cal_shannon_ent(sub_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_gain:
            best_gain, best_feature = info_gain, i
    return best_feature

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    return max(class_count, key=class_count.get)

def creat_tree(dataset, labels):
    class_list = [vec[-1] for vec in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature(dataset)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_vals = [vec[best_feat] for vec in dataset]
    unique_vals = set(feat_vals)
    for val in unique_vals:
        sub_labels = labels.copy()
        tree[best_feat_label][val] = creat_tree(
            split_dataset(dataset, best_feat, val), sub_labels
        )
    return tree

# ---------- 2. 可视化函数 ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(ax, txt, center, parent, node_type):
    ax.annotate(txt, xy=parent, xycoords='axes fraction',
                xytext=center, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args)

def get_leaf_count(tree):
    count = 0
    first_key = next(iter(tree))
    child_dict = tree[first_key]
    for child in child_dict.values():
        if isinstance(child, dict):
            count += get_leaf_count(child)
        else:
            count += 1
    return count

def get_tree_depth(tree):
    depth = 0
    first_key = next(iter(tree))
    child_dict = tree[first_key]
    for child in child_dict.values():
        if isinstance(child, dict):
            this_depth = 1 + get_tree_depth(child)
        else:
            this_depth = 1
        depth = max(depth, this_depth)
    return depth

def plot_mid_text(ax, center, parent, txt):
    x_mid = (parent[0] + center[0]) / 2.0
    y_mid = (parent[1] + center[1]) / 2.0
    ax.text(x_mid, y_mid, txt, va="center", ha="center", fontsize=10)

def plot_tree(ax, tree, parent, node_txt, total_leaf, total_depth, offset):
    leaf_count = get_leaf_count(tree)
    first_key = next(iter(tree))
    center = (
        offset['x'] + (1.0 + leaf_count) / (2.0 * total_leaf),
        offset['y']
    )
    if node_txt:
        plot_mid_text(ax, center, parent, node_txt)
    plot_node(ax, first_key, center, parent, decision_node)
    child_dict = tree[first_key]
    offset['y'] -= 1.0 / total_depth
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center, str(key), total_leaf, total_depth, offset)
        else:
            offset['x'] += 1.0 / total_leaf
            leaf_center = (offset['x'], offset['y'])
            plot_node(ax, str(child), leaf_center, center, leaf_node)
            plot_mid_text(ax, leaf_center, center, str(key))
    offset['y'] += 1.0 / total_depth

def create_plot(tree):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    total_leaf = get_leaf_count(tree)
    total_depth = get_tree_depth(tree)
    offset = {'x': -0.5 / total_leaf, 'y': 1.0}
    plot_tree(ax, tree, (0.5, 1.0), '', total_leaf, total_depth, offset)
    plt.show()

# ---------- 3. 评估函数 ----------
def classify(tree, labels, vec):
    first_key = next(iter(tree))
    child_dict = tree[first_key]
    feat_idx = labels.index(first_key)
    for key in child_dict:
        if vec[feat_idx] == key:
            if isinstance(child_dict[key], dict):
                return classify(child_dict[key], labels, vec)
            else:
                return child_dict[key]

def calculate_accuracy(tree, labels, dataset):
    correct = 0
    total = len(dataset)
    for vec in dataset:
        true_label = vec[-1]
        pred_label = classify(tree, labels, vec[:-1])
        if pred_label == true_label:
            correct += 1
    return correct / total

# ---------- 4. 数据集加载 ----------
def load_lenses_data():
    with open('lenses.txt', 'r') as f:
        return [line.strip().split('\t') for line in f]

# ---------- 5. 主运行入口 ----------
if __name__ == "__main__":
    # 1. 加载数据
    dataset = load_lenses_data()
    labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    
    # 2. 训练决策树
    tree = creat_tree(dataset, labels.copy())
    print("决策树结构:", tree)
    
    # 3. 可视化决策树
    create_plot(tree)
    
    # 4. 计算准确率
    accuracy = calculate_accuracy(tree, ['age', 'prescription', 'astigmatic', 'tear_rate'], dataset)
    print(f"训练集准确率: {accuracy:.2%}")