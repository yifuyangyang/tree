from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

# -------------------------- 1. 熵计算函数 --------------------------
def cal_shannon_ent(dataset):
    """计算数据集的香农熵（衡量不确定性）"""
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

# -------------------------- 2. 数据集划分函数 --------------------------
def split_dataset(dataset, axis, value):
    """按指定特征（axis）的取值（value）划分数据集，返回去掉该特征的子数据集"""
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

# -------------------------- 3. 最优特征选择函数 --------------------------
def choose_best_feature_split(dataset):
    """选择信息增益最大的特征作为最优划分特征"""
    num_features = len(dataset[0])-1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = 0
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

# -------------------------- 4. 多数投票函数 --------------------------
def majority_cnt(class_list):
    """统计类别出现次数，返回多数类标签"""
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

# -------------------------- 5. 决策树构建函数 --------------------------
def creat_tree(dataset, labels):
    """递归构建决策树，返回树的字典结构"""
    class_list = [example[-1] for example in dataset]
    # 递归出口1：所有样本属于同一类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归出口2：无特征可分，返回多数类
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征
    best_feat = choose_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])  # 删除已使用的特征名称
    # 遍历特征的所有取值，递归构建子树
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]  # 拷贝特征名称，避免递归修改原列表
        my_tree[best_feat_label][value] = creat_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree

# -------------------------- 6. 可视化相关函数 --------------------------
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 节点样式定义
decision_node = dict(boxstyle="sawtooth", fc='0.8')  # 决策节点：锯齿边框
leaf_node = dict(boxstyle="round4", fc='0.8')        # 叶节点：圆角边框
arrow_args = dict(arrowstyle="<-")                   # 箭头样式：指向父节点

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    """绘制节点（决策节点/叶节点）和箭头"""
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    """获取决策树的叶节点数量（用于计算可视化宽度）"""
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
    """获取决策树的深度（用于计算可视化高度）"""
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
    """在箭头中间添加文字（特征取值）"""
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    """递归绘制决策树"""
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]
    num_leafs = get_num_leafs(my_tree)
    
    # 计算当前节点中心位置
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), x_off_y['y_off'])
    # 绘制箭头中间文字
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    # 绘制决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    
    # 递归绘制子树
    x_off_y['y_off'] -= 1.0 / total_d  # 向下移动一层
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 绘制叶节点
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    x_off_y['y_off'] += 1.0 / total_d  # 回溯到上一层

def create_plot(my_tree):
    """创建决策树可视化图表"""
    fig, ax = plt.subplots(figsize=(10, 8))  # 设置图表大小
    ax.set_axis_off()  # 隐藏坐标轴
    total_w = float(get_num_leafs(my_tree))  # 树的宽度（叶节点数量）
    total_d = float(get_tree_depth(my_tree))  # 树的高度（深度）
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}  # 初始位置偏移
    # 递归绘制树
    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)
    plt.tight_layout()
    plt.show()

# -------------------------- 7. 预测函数 --------------------------
def classify(input_tree, feat_labels, test_vec):
    """
    使用构建好的决策树进行预测
    参数：
        input_tree: 决策树字典
        feat_labels: 特征名称列表（与测试数据顺序一致）
        test_vec: 测试样本（特征值列表，无标签）
    返回：
        class_label: 预测标签
    """
    # 获取根节点的特征名称
    first_str = next(iter(input_tree))
    # 获取根节点的子树字典
    second_dict = input_tree[first_str]
    # 找到根节点特征在特征列表中的索引
    feat_index = feat_labels.index(first_str)
    # 遍历子树的所有取值
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            # 如果子节点是字典（决策节点），递归预测
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            # 如果子节点是叶节点，直接返回标签
            else:
                class_label = second_dict[key]
    return class_label

# -------------------------- 8. 数据加载与预处理 --------------------------
def load_lenses_data():
    """加载 lenses.txt 数据集"""
    # 读取文件
    with open('lenses.txt', 'r') as f:
        lines = f.readlines()
    # 处理每一行数据
    dataset = []
    for line in lines:
        # 去除换行符，按制表符分割
        line = line.strip().split('\t')
        dataset.append(line)
    # 特征名称
    labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    return dataset, labels

# -------------------------- 9. 模型评估 --------------------------
def calculate_accuracy(input_tree, feat_labels, dataset):
    """计算模型在数据集上的准确率"""
    correct_count = 0
    total_count = len(dataset)
    for sample in dataset:
        # 提取特征值（去掉最后一列标签）
        test_vec = sample[:-1]
        # 真实标签
        true_label = sample[-1]
        # 预测标签
        pred_label = classify(input_tree, feat_labels, test_vec)
        # 判断是否预测正确
        if pred_label == true_label:
            correct_count += 1
    # 计算准确率
    accuracy = correct_count / total_count
    return accuracy

# -------------------------- 10. 主函数 --------------------------
if __name__ == "__main__":
    # 1. 加载数据
    dataset, labels = load_lenses_data()
    print("数据集大小：", len(dataset))
    print("特征名称：", labels)
    print("前5条数据：", dataset[:5])
    
    # 2. 训练决策树（注意：传入特征名称的拷贝，避免原列表被修改）
    tree = creat_tree(dataset, labels[:])
    print("\n决策树结构：", tree)
    print("决策树叶节点数：", get_num_leafs(tree))
    print("决策树深度：", get_tree_depth(tree))
    
    # 3. 可视化决策树
    create_plot(tree)
    
    # 4. 计算训练集准确率
    accuracy = calculate_accuracy(tree, ['age', 'prescription', 'astigmatic', 'tear_rate'], dataset)
    print(f"\n训练集准确率：{accuracy:.2%}")