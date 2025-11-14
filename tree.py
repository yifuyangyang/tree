from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def cal_shannon_ent(dataset):
    """计算香农熵（添加空数据集防护）"""
    num_entries = len(dataset)
    if num_entries == 0:
        return 0.0
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

def create_dataSet():
    """创建示例数据集"""
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels

# 测试熵计算
dataset, labels = create_dataSet()
print("示例数据集熵值:", cal_shannon_ent(dataset))

def split_dataset(dataset, axis, value):
    """按特征划分数据集（去除该特征列）"""
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

# 测试数据集划分
dataset_test = [
    [1, 'sunny', 'yes'],
    [1, 'rainy', 'no'],
    [0, 'sunny', 'yes']
]
result = split_dataset(dataset_test, 0, 1)
print("按特征0=1划分结果:", result)

def choose_best_feature_split(dataset):
    """选择信息增益最大的特征索引（添加边界校验）"""
    num_samples = len(dataset)
    if num_samples == 0:  # 空数据集防护
        return 0
    num_features = len(dataset[0]) - 1  # 最后一列是标签
    if num_features == 0:  # 无特征可分
        return 0
    
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = -1.0  # 初始化改为-1（避免信息增益为0时不更新）
    best_feature = 0
    
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/float(num_samples)
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        # 信息增益更大时更新最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    """统计类别出现次数，返回出现最多的类别"""
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 按出现次数降序排序，返回第一个（最多）
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def creat_tree(dataset, labels):
    """创建决策树（修复：不原地修改labels，添加数据校验）"""
    # 数据校验：确保dataset和labels非空
    if len(dataset) == 0 or len(labels) == 0:
        return majority_cnt([example[-1] for example in dataset] if dataset else ['no lenses'])
    
    class_list = [example[-1] for example in dataset]
    
    # 递归出口1：所有样本同类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 递归出口2：无特征可分，返回多数类
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    # 选择最优特征（添加索引校验）
    best_feat = choose_best_feature_split(dataset)
    # 确保best_feat在labels索引范围内
    if best_feat >= len(labels):
        best_feat = 0
    best_feat_label = labels[best_feat]
    
    # 构建决策树
    my_tree = {best_feat_label: {}}
    
    # 提取该特征的所有唯一值
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    
    # 递归构建子树（传入标签拷贝，不修改原始labels）
    for value in unique_vals:
        # 拷贝标签列表，排除当前最优特征
        sub_labels = labels[:best_feat] + labels[best_feat+1:]
        # 划分数据集并递归
        my_tree[best_feat_label][value] = creat_tree(
            split_dataset(dataset, best_feat, value),
            sub_labels
        )
    return my_tree

def classify(input_tree, feat_labels, test_vec):
    """使用决策树分类"""
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    # 找到特征名对应的索引（feat_labels是原始完整标签列表）
    feat_index = feat_labels.index(first_str)
    
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label
    return 'no lenses'  # 无匹配特征值时返回默认类别

def calculate_accuracy(tree, dataset, feat_labels):
    """计算准确率"""
    correct_count = 0
    total_count = len(dataset)
    if total_count == 0:
        return 0.0
    for i in range(total_count):
        test_vec = dataset[i][:-1]  # 测试样本特征（不含标签）
        true_label = dataset[i][-1]  # 真实标签
        predicted_label = classify(tree, feat_labels, test_vec)
        if predicted_label == true_label:
            correct_count += 1
    return correct_count / total_count

# ---------------------- 绘图相关函数 ----------------------
decision_node = dict(boxstyle="sawtooth", fc='0.8')
leaf_node = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    """获取叶节点数量"""
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
    """获取树的深度"""
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
    """在箭头中间添加文字"""
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
    
    # 绘制箭头中间文字
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
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()  # 隐藏坐标轴
    
    # 计算树的叶节点数和深度
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    
    # 初始化坐标偏移
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}
    
    # 绘制决策树
    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)
    
    plt.tight_layout()
    plt.show()

# ---------------------- 加载lenses数据集并运行 ----------------------
def load_data(filepath):
    """加载lenses数据集（修复：按制表符分隔，处理空行）"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as fr:
            for line_num, line in enumerate(fr, 1):  # 记录行号，方便排查错误
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 关键修复：按制表符分隔（原数据是tab分隔，不是空格）
                parts = line.split('\t')
                # 校验每行数据是否有5列（4特征+1标签）
                if len(parts) != 5:
                    print(f"警告：第{line_num}行数据格式错误，跳过该行！内容：{line}")
                    continue
                data.append(parts)
        print(f"成功加载 {len(data)} 条有效数据")
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}，请检查文件路径是否正确！")
        exit(1)
    return data

# 数据集路径（确保lenses.txt在当前目录，或改为绝对路径）
lenspath = 'lenses.txt'  # 若报错，改为绝对路径：r'C:\Users\asus\Desktop\gwt\tree\lenses.txt'

# 加载数据和标签
labels_lenses = ['年龄', '屈光', '散光', '泪液分泌']
dataset = load_data(lenspath)

# 数据校验：确保数据集非空
if len(dataset) == 0:
    print("错误：未加载到有效数据，请检查文件内容！")
    exit(1)

# 构建决策树（传入标签拷贝）
tree = creat_tree(dataset, labels_lenses[:])
print("\n决策树结构:")
print(tree)

# 计算准确率
accuracy = calculate_accuracy(tree, dataset, labels_lenses)
print(f"\n训练集准确率: {accuracy*100:.2f}%")

# 可视化决策树
create_plot(tree)