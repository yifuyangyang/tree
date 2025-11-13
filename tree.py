from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

# 支持中文显示（解决中文乱码问题）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文支持
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

def cal_shannon_ent(dataset):
    """计算数据集的香农熵（量化不确定性）"""
    num_entries = len(dataset)
    labels_counts = {}
    # 统计每个类别标签的出现次数
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1
    # 计算香农熵
    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_dataSet():
    """创建示例数据集（用于测试熵计算）"""
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],  # 修复原数据格式错误（少一个特征值）
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # 修正拼写错误（suerfacing→surfacing）
    return dataset, labels

def split_dataset(dataset, axis, value):
    """按指定特征和取值划分数据集（剔除该特征列）"""
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_split(dataset):
    """选择信息增益最大的最优划分特征（返回特征索引）"""
    num_features = len(dataset[0]) - 1  # 最后一列是标签，不算特征
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 0  # 修正初始值（原1改为0，符合索引逻辑）
    # 遍历每个特征计算信息增益
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)  # 特征的唯一取值
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        # 更新最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    """统计类别出现次数，返回按次数降序排列的类别列表"""
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 按出现次数降序排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count

def create_tree(dataset, labels):
    """递归构建决策树"""
    class_list = [example[-1] for example in dataset]
    # 递归出口1：所有样本属于同一类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归出口2：无特征可分，返回多数类
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)[0][0]  # 返回出现次数最多的类别
    # 选择最优划分特征
    best_feat = choose_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    # 构建决策树（字典结构）
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])  # 剔除已使用的特征标签
    # 遍历特征的所有唯一取值，递归构建子树
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]  # 拷贝标签列表（避免递归修改原列表）
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree

# ---------------------- 决策树可视化相关函数 ----------------------
def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    """绘制决策节点或叶节点"""
    # 节点样式定义
    decision_node = dict(boxstyle="sawtooth", fc='0.8')  # 决策节点（锯齿框）
    leaf_node = dict(boxstyle="round4", fc='0.8')       # 叶节点（圆角框）
    arrow_args = dict(arrowstyle="<-")                  # 箭头样式（子→父）
    # 绘制带箭头的节点
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    """统计决策树的叶子节点数量（用于布局计算）"""
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):  # 子节点是字典→递归统计
            num_leafs += get_num_leafs(second_dict[key])
        else:  # 子节点是类别→叶子节点
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    """统计决策树的深度（用于布局计算）"""
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):  # 子节点是字典→递归计算深度
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:  # 子节点是类别→深度+1
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    """在父节点和子节点之间添加文字（标注特征取值）"""
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    """递归绘制决策树"""
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]
    # 计算当前节点位置（基于叶子数和深度的布局）
    num_leafs = get_num_leafs(my_tree)
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), x_off_y['y_off'])
    # 绘制父→子的特征取值标注
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    # 绘制当前决策节点
    decision_node = dict(boxstyle="sawtooth", fc='0.8')
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    # 递归绘制子树
    x_off_y['y_off'] -= 1.0 / total_d  # 向下移动一层
    for key, child in child_dict.items():
        if isinstance(child, dict):  # 子节点是树→递归绘制
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:  # 子节点是叶子→绘制叶节点
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            leaf_node = dict(boxstyle="round4", fc='0.8')
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    x_off_y['y_off'] += 1.0 / total_d  # 回溯到上一层

def create_plot(my_tree):
    """创建决策树可视化图形并显示"""
    fig, ax = plt.subplots(figsize=(10, 6))  # 设置图形大小
    ax.set_axis_off()  # 隐藏坐标轴
    # 计算树的叶子数和深度（用于布局）
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}  # 初始位置偏移
    # 绘制决策树
    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)
    plt.tight_layout()  # 自动调整布局
    plt.show()

# ---------------------- 主程序：建树 + 可视化 ----------------------
if __name__ == "__main__":
    # 1. 数据集（天气与打球决策数据）
    weather_data = [
        ['Sunny', 'Hot', 'High', False, 'No'],
        ['Sunny', 'Hot', 'High', True, 'No'],
        ['Overcast', 'Hot', 'High', False, 'Yes'],
        ['Rain', 'Mild', 'High', False, 'Yes'],
        ['Rain', 'Cool', 'Normal', False, 'Yes'],
        ['Rain', 'Cool', 'Normal', True, 'No'],
        ['Overcast', 'Cool', 'Normal', True, 'Yes'],
        ['Sunny', 'Mild', 'High', False, 'No'],
        ['Sunny', 'Cool', 'Normal', False, 'Yes'],
        ['Rain', 'Mild', 'Normal', False, 'Yes'],
        ['Sunny', 'Mild', 'Normal', True, 'Yes'],
        ['Overcast', 'Mild', 'High', True, 'Yes'],
        ['Overcast', 'Hot', 'Normal', False, 'Yes'],
        ['Rain', 'Mild', 'High', True, 'No']
    ]
    # 2. 特征标签（与数据集特征列一一对应）
    labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    # 3. 构建决策树（传入标签拷贝，避免原列表被修改）
    tree = create_tree(weather_data, labels[:])
    # 4. 可视化决策树
    create_plot(tree)