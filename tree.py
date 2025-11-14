from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

# ===================== 中文乱码修复（Windows系统）=====================
import sys
# 设置标准输出编码为UTF-8（解决控制台中文乱码）
sys.stdout.reconfigure(encoding='utf-8')
# 设置matplotlib中文（兼容所有Windows版本）
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei']  # 多字体兼容
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== ID3决策树核心函数（保持你原来的实现）=====================
def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1
    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key])/num_entries
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_split(dataset):
    num_features = len(dataset[0])-1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 0  # 修正初始值（原1可能导致索引异常）
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        info_gain = base_entropy-new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys():class_count[vote]=0
        class_count[vote]+=1
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count

def creat_tree(dataset,labels):
    class_list=[example[-1] for example in dataset]
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
    if len(dataset[0])==1:
        return majority_cnt(class_list)[0][0]  # 直接返回多数标签（避免返回列表）
    best_feat=choose_best_feature_split(dataset)
    best_feat_label=labels[best_feat]
    my_tree={best_feat_label:{}}
    del(labels[best_feat])
    feat_values=[example[best_feat] for example in dataset]
    unique_vals=set(feat_values)
    for value in unique_vals:
        sub_labels=labels[:]
        my_tree[best_feat_label][value]=creat_tree(split_dataset(dataset,best_feat,value),sub_labels)
    return my_tree

# ===================== 绘图相关函数（保持你原来的实现）=====================
decision_node=dict(boxstyle="sawtooth",fc='0.8')
leaf_node=dict(boxstyle="round4",fc='0.8')
arrow_args=dict(arrowstyle="<-")

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    num_leafs = 0
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    max_depth = 0
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]
    num_leafs = get_num_leafs(my_tree)
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), x_off_y['y_off'])
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    x_off_y['y_off'] += 1.0 / total_d

def create_plot(my_tree):
    fig, ax = plt.subplots(figsize=(10, 6))  # 调整图大小，避免节点重叠
    ax.set_axis_off()
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}
    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)
    plt.tight_layout()
    plt.show()

# ===================== 新增：数据集加载、预测、评估函数=====================
def load_lenses_data():
    """加载lenses.txt数据集"""
    dataset = []
    with open('lenses.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = line.split() if '\t' not in line else line.split('\t')
            dataset.append(sample)
    feature_names = ['age', 'prescription', 'astigmatic', 'tear_rate']
    return dataset, feature_names

def predict_sample(tree_model, feature_names, sample):
    """对单个样本预测"""
    current_node = tree_model
    while isinstance(current_node, dict):
        feature_name = next(iter(current_node.keys()))
        feature_idx = feature_names.index(feature_name)
        sample_feature_val = sample[feature_idx]
        if sample_feature_val in current_node[feature_name]:
            current_node = current_node[feature_name][sample_feature_val]
        else:
            child_nodes = current_node[feature_name]
            labels = []
            for val, node in child_nodes.items():
                if not isinstance(node, dict):
                    labels.append(node)
            return majority_cnt(labels)[0][0]
    return current_node

def calculate_accuracy(tree_model, feature_names, dataset):
    """计算训练集准确率"""
    correct_count = 0
    total_count = len(dataset)
    for sample in dataset:
        sample_features = sample[:-1]
        true_label = sample[-1]
        pred_label = predict_sample(tree_model, feature_names, sample_features)
        if pred_label == true_label:
            correct_count += 1
    return round(correct_count / total_count, 4) if total_count > 0 else 0.0

# ===================== 主函数（程序入口）=====================
def main():
    # 1. 加载数据集
    dataset, feature_names = load_lenses_data()
    print(f"数据集样本数：{len(dataset)}")
    print(f"特征名称：{feature_names}")
    print(f"前3个样本：{dataset[:3]}\n")
    
    # 2. 训练ID3决策树（传入特征名称拷贝，避免原列表被修改）
    tree_model = creat_tree(dataset, feature_names.copy())  # 直接调用本地函数，无模块冲突
    print(f"训练好的决策树：\n{tree_model}\n")
    
    # 3. 计算准确率
    accuracy = calculate_accuracy(tree_model, feature_names, dataset)
    print(f"训练集准确率：{accuracy * 100:.2f}%\n")
    
    # 4. 绘制决策树（截图保存）
    create_plot(tree_model)

# ===================== 运行程序 =====================
if __name__ == "__main__":
    main()
