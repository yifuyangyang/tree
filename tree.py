from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib


def cal_shannon_ent(dataset):
    """
    计算数据集的信息熵（香农熵）
    """
    num_entries = len(dataset)
    label_counts = {}
    
    # 统计每个类别标签出现的次数
    for feat_vec in dataset:
        current_label = feat_vec[-1]  # 获取类别标签
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    
    # 计算香农熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    
    return shannon_ent


def create_dataSet():
    """
    创建示例数据集
    熵接近1：两个类别的比例接近，数据集不确定性高
    熵接近0：类别集中，数据集确定性高
    """
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    """
    按照指定特征的某个取值划分数据集
    
    参数：
        dataset: 原始数据集
        axis: 特征列索引
        value: 特征的目标取值
    
    返回：
        ret_dataset: 划分后的子数据集（不包含axis特征列）
    """
    ret_dataset = []
    
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            # 移除指定特征列
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    
    return ret_dataset


def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征作为划分特征
    
    参数：
        dataset: 数据集
    
    返回：
        best_feature: 最优特征的索引
    """
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    for i in range(num_features):
        # 获取该特征的所有取值
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        
        # 计算该特征的条件熵
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        
        # 计算信息增益
        info_gain = base_entropy - new_entropy
        
        # 更新最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    
    return best_feature


def majority_cnt(class_list):
    """
    统计类别列表中各类别的出现次数，按次数降序排列
    
    参数：
        class_list: 类别列表
    
    返回：
        按出现次数降序排列的类别计数列表
    """
    class_count = {}
    
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    
    # 按出现次数降序排序
    sorted_class_count = sorted(class_count.items(), 
                               key=operator.itemgetter(1), 
                               reverse=True)
    return sorted_class_count


def creat_tree(dataset, labels):
    """
    递归构建决策树
    """
    class_list = [example[-1] for example in dataset]
    
    # 递归终止条件1：所有样本属于同一类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 递归终止条件2：没有更多特征可供划分
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    # 选择最优划分特征
    best_feat = choose_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    
    # 构建决策树节点
    my_tree = {best_feat_label: {}}
    del labels[best_feat]  # 移除已使用的特征标签
    
    # 获取该特征的所有唯一取值
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    
    # 递归构建子树
    for value in unique_vals:
        sub_labels = labels[:]  # 创建标签副本
        my_tree[best_feat_label][value] = creat_tree(
            split_dataset(dataset, best_feat, value), 
            sub_labels
        )
    
    return my_tree


# 配置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义节点样式
decision_node = dict(boxstyle="sawtooth", fc='0.8')
leaf_node = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")


def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    """
    绘制决策树节点
    """
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')


def get_num_leafs(my_tree):
    """
    计算决策树的叶子节点数量
    """
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
    """
    计算决策树的深度
    """
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    max_depth = 0
    
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        max_depth = max(max_depth, this_depth)
    
    return max_depth


def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    """
    在父子节点之间绘制文本
    """
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)


def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    """
    递归绘制决策树
    """
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]

    num_leafs = get_num_leafs(my_tree)
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), 
                 x_off_y['y_off'])

    # 绘制边上的文本
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)

    # 绘制决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)

    # 进入下一层
    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 绘制叶子节点
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    
    # 返回上一层
    x_off_y['y_off'] += 1.0 / total_d


def create_plot(my_tree):
    """
    创建决策树可视化
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}

    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)

    plt.tight_layout()
    plt.show()


# ========== 示例使用 ==========
if __name__ == "__main__":
    # 测试香农熵计算
    dataset, labels = create_dataSet()
    print("数据集香农熵:", cal_shannon_ent(dataset))
    
    # 测试数据集划分
    dataset_test = [
        [1, 'sunny', 'yes'],
        [1, 'rainy', 'no'],
        [0, 'sunny', 'yes']
    ]
    result = split_dataset(dataset_test, 0, 1)
    print("划分结果:", result)
    
    # 天气数据集示例
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
    
    feature_labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    
    # 生成决策树
    decision_tree = creat_tree(weather_data, feature_labels[:])
    print("生成的决策树:", decision_tree)
    
    # 可视化决策树
    create_plot(decision_tree)
