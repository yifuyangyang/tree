from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib



# -------------------------- 1. 基础工具函数（熵计算、数据集划分等）--------------------------
def cal_shannon_ent(dataset):
    """计算数据集的香农熵"""
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts:
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1
    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def split_dataset(dataset, axis, value):
    """按指定特征和取值划分数据集（移除该特征列）"""
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_split(dataset):
    """选择信息增益最大的最优特征（返回索引）"""
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 0  # 修复：初始化为0（避免索引越界）
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    """投票法返回多数类（用于特征耗尽时）"""
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 修复：返回多数类标签（而非列表）

def creat_tree(dataset, labels):
    """递归构建ID3决策树"""
    class_list = [example[-1] for example in dataset]
    # 递归出口1：所有样本同类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归出口2：无特征可分，返回多数类
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征并构建树
    best_feat = choose_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = creat_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree

# -------------------------- 2. 决策树可视化函数 --------------------------
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 节点样式定义
decision_node = dict(boxstyle="sawtooth", fc='0.8')
leaf_node = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    """绘制决策节点/叶节点"""
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    """统计叶子节点数（用于布局）"""
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    """统计树深度（用于布局）"""
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    """在箭头中间添加特征取值文本"""
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    """递归绘制决策树"""
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]
    num_leafs = get_num_leafs(my_tree)
    # 计算当前节点位置
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), x_off_y['y_off'])
    # 绘制箭头中间文本
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    # 绘制决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    # 递归绘制子树
    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 绘制叶节点
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    x_off_y['y_off'] += 1.0 / total_d

def create_plot(my_tree):
    """创建决策树可视化图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}
    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)
    plt.tight_layout()
    plt.show()

# -------------------------- 3. 新增核心功能：数据加载、预测、评估 --------------------------
def load_lenses_data(file_path):
    """加载lenses.txt数据集"""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()  # 去除换行符和空格
            if not line:
                continue  # 跳过空行
            # 按制表符分割（数据格式：特征1\t特征2\t特征3\t特征4\t标签）
            sample = line.split('\t')
            dataset.append(sample)
    # 特征名称（与数据列对应）
    labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    return dataset, labels

def predict(tree, feature_labels, sample):
    """使用决策树预测单个样本"""
    # 获取根节点（第一个特征）
    root_node = next(iter(tree))
    # 根节点对应的子树
    child_tree = tree[root_node]
    # 找到根节点特征在样本中的索引
    feature_idx = feature_labels.index(root_node)
    # 遍历子树，匹配样本特征取值
    for key in child_tree:
        if sample[feature_idx] == key:
            # 如果子节点是字典（决策节点），递归预测
            if isinstance(child_tree[key], dict):
                return predict(child_tree[key], feature_labels, sample)
            # 如果是叶节点，返回标签
            else:
                return child_tree[key]
    # 若未匹配到（异常情况），返回默认标签
    return 'no lenses'

def evaluate(tree, feature_labels, dataset):
    """评估模型在数据集上的准确率"""
    correct_count = 0
    total_count = len(dataset)
    for sample in dataset:
        # 分离样本特征（前4列）和真实标签（最后1列）
        sample_features = sample[:-1]
        true_label = sample[-1]
        # 预测
        pred_label = predict(tree, feature_labels, sample_features)
        # 统计正确数
        if pred_label == true_label:
            correct_count += 1
    # 计算准确率
    accuracy = correct_count / total_count
    return accuracy, correct_count, total_count

# -------------------------- 4. 主程序：训练+预测+评估+可视化 --------------------------
if __name__ == "__main__":
    # 1. 加载数据（请确保lenses.txt与代码在同一目录）
    file_path = 'lenses.txt'  # 数据集文件路径
    dataset, feature_labels = load_lenses_data(file_path)
    print(f"数据集加载完成，共{len(dataset)}个样本")
    print(f"特征名称：{feature_labels}")
    print(f"前3个样本：{dataset[:3]}")

    # 2. 训练ID3决策树（传入特征标签拷贝，避免原列表被修改）
    tree = creat_tree(dataset, feature_labels[:])
    print("\n生成的决策树：")
    print(tree)

    # 3. 可视化决策树
    print("\n正在绘制决策树...")
    create_plot(tree)

    # 4. 模型评估（使用训练集评估准确率）
    accuracy, correct, total = evaluate(tree, ['age', 'prescription', 'astigmatic', 'tear_rate'], dataset)
    print(f"\n模型评估结果：")
    print(f"总样本数：{total}")
    print(f"正确预测数：{correct}")
    print(f"训练集准确率：{accuracy:.2%}")
