from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

def cal_shannon_ent(dataset):
    """
    计算熵
    """
    # 1. 计算数据集中样本的总数
    num_entries = len(dataset)
    if num_entries == 0:
        return 0
    
    # 2. 创建一个字典，用于统计每个类别标签出现的次数
    labels_counts = {}
    # 3. 遍历数据集中的每条记录
    for feat_vec in dataset:
        # feat_vec[-1] 表示每条样本的最后一个元素 类别标签
        current_label = feat_vec[-1]
        # 如果该标签是第一次出现，则在字典中初始化为 0
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        # 累加该标签出现的次数
        labels_counts[current_label] += 1

    # 4. 计算香农熵
    shannon_ent = 0.0
    # 遍历字典中的每个类别及其计数
    for key in labels_counts:
        # 计算该类别的概率
        prob = float(labels_counts[key])/num_entries
        # 根据香农熵公式累加：
        shannon_ent -= prob * log(prob, 2)
    # 5. 返回计算得到的熵值
    return shannon_ent


def create_dataSet():
    """
    熵接近 1，说明"yes"和"no"两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0,类别越集中，数据集越"纯"或"确定性越强"
    """
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    """
    按照指定特征(axis)的某个取值(value)划分数据集。
    会选出所有该特征等于 value 的样本，
    并且返回时会去掉这一列特征。

    参数：
        dataset: 原始数据集（二维列表，每一行是一个样本，每一列是一个特征，最后一列通常是标签）
        axis: 要划分的特征列索引（例如 0 表示第 1 个特征）
        value: 特征的目标取值（例如 'sunny'）

    返回：
        ret_dataset: 划分后的子数据集（不包含 axis 那一列）
    """
    ret_dataset = []  # 用于存放划分后的子数据集
    # 遍历原始数据集的每一条样本
    for feat_vec in dataset:
        # 如果这一条样本在 axis 特征上的值等于给定的 value
        if feat_vec[axis] == value:
            # 构建一个"去掉该特征"的新样本
            reduced_feat_vec = feat_vec[:axis]    # 取前面部分
            reduced_feat_vec.extend(feat_vec[axis+1:])  # 取后面部分拼接起来
            # 把这个新样本加入到子数据集中
            ret_dataset.append(reduced_feat_vec)
    # 返回划分后的数据集
    return ret_dataset


def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征索引，作为本轮划分的最优特征。

    参数：
        dataset: 数据集（二维列表，每行一条样本，最后一列是标签）
    返回：
        best_feature: 最优特征的索引位置
    """
    # 1. 计算特征总数（最后一列是标签，不算特征）
    num_features = len(dataset[0]) - 1
    # 2. 计算原始数据集的熵（未划分前的不确定性）
    base_entropy = cal_shannon_ent(dataset)
    # 3. 初始化"最大信息增益"和"最佳特征"
    best_info_gain = 0.0
    best_feature = -1  # 初始化为-1，表示没有找到合适的特征
    
    # 4. 遍历每一个特征，计算它的信息增益
    for i in range(num_features):
        # 4.1 提取出该特征所有样本的取值列表
        feat_list = [example[i] for example in dataset]
        # 4.2 获取该特征的所有唯一取值,转换为set集合，自动去重
        unique_val = set(feat_list)
        # 4.3 计算该特征划分后的"加权平均熵"
        new_entropy = 0.0
        for value in unique_val:
            # 按照该特征的某个取值划分数据集
            sub_dataset = split_dataset(dataset, i, value)
            # 计算该子集占整个数据集的比例
            prob = len(sub_dataset) / float(len(dataset))
            # 累加加权熵（概率 * 子集熵）
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        # 4.4 计算该特征的信息增益
        info_gain = base_entropy - new_entropy
        # 4.5 如果当前特征信息增益更大，就更新最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    
    # 如果所有特征的信息增益都为0，返回第一个特征
    if best_feature == -1:
        best_feature = 0
        
    return best_feature


def majority_cnt(class_list):
    """
    功能：统计 class_list 中各类别出现的次数，并按出现次数从多到少排序返回。
    参数：
        class_list: 列表，例如 ['yes', 'no', 'yes', 'yes', 'no']
    返回：
        出现次数最多的类别
    """
    # 1. 定义一个空字典，用于存放每个类别及其计数
    class_count = {}
    # 2. 遍历类别列表，对每个类别进行计数
    for vote in class_list:
        # 如果该类别还未在字典中出现，先初始化计数为0
        if vote not in class_count.keys():
            class_count[vote] = 0
        # 累加该类别的出现次数
        class_count[vote] += 1
    # 3. 将字典的键值对（类别, 次数）转为列表，并按次数进行降序排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的类别
    return sorted_class_count[0][0]


def creat_tree(dataset, labels):
    """
    创建决策树
    """
    # 取出数据集中每条样本的"标签列"（通常是最后一列）
    class_list = [example[-1] for example in dataset]
    
    # 递归出口①：若所有样本同类，直接返回该类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 递归出口②：若没有可用特征（只剩标签列），返回多数类
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    # 检查labels是否为空
    if not labels:
        return majority_cnt(class_list)
    
    # 选择"最优划分特征"的下标
    best_feat = choose_best_feature_split(dataset)
    
    # 确保 best_feat 在有效范围内
    if best_feat >= len(labels):
        return majority_cnt(class_list)
        
    best_feat_label = labels[best_feat]
    
    # 构建当前节点
    my_tree = {best_feat_label: {}}
    
    # 保存当前特征的标签，然后从labels中删除
    current_label = labels[best_feat]
    del(labels[best_feat])
    
    # 取出该特征在所有样本上的取值列表
    feat_values = [example[best_feat] for example in dataset]
    # 去重：该特征有哪些不同的取值
    unique_vals = set(feat_values)
    
    # 对该特征的每个取值，分别递归构建子树
    for value in unique_vals:
        sub_labels = labels[:]   # 拷贝一份标签名列表给子递归使用
        # 把当前特征=某取值的样本切分出来
        sub_dataset = split_dataset(dataset, best_feat, value)
        # 如果子数据集为空，返回多数类
        if len(sub_dataset) == 0:
            my_tree[best_feat_label][value] = majority_cnt(class_list)
        else:
            my_tree[best_feat_label][value] = creat_tree(sub_dataset, sub_labels)
    
    return my_tree


# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义节点样式
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
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
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

    # 边文字（父->子取值）
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)

    # 决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)

    # 进入下一层
    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 叶子
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    # 返回上一层
    x_off_y['y_off'] += 1.0 / total_d


def create_plot(my_tree):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    
    if total_w == 0:
        total_w = 1
    if total_d == 0:
        total_d = 1
        
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}

    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)

    plt.tight_layout()
    plt.show()


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类预测
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    
    try:
        feat_index = feat_labels.index(first_str)
    except ValueError:
        # 如果特征不在标签列表中，返回None
        return None
    
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                return classify(second_dict[key], feat_labels, test_vec)
            else:
                return second_dict[key]
    
    # 如果没有匹配的特征值，返回None
    return None


def load_lenses(path):
    """
    加载隐形眼镜数据集
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts and len(parts) >= 5:  # 确保有足够的列
                data.append(parts)
    return data


# 主程序
if __name__ == "__main__":
    # 加载数据
    try:
        lenses_data = load_lenses('lenses.txt')
        print(f"成功加载 {len(lenses_data)} 条数据")
        
        lenses_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
        
        # 创建决策树
        lenses_tree = creat_tree(lenses_data, lenses_labels[:])
        print("决策树构建完成!")
        print("决策树结构:", lenses_tree)
        
        # 绘制决策树
        create_plot(lenses_tree)
        
        # 计算准确率
        correct = 0
        feature_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
        for row in lenses_data:
            x, y = row[:-1], row[-1]
            yhat = classify(lenses_tree, feature_labels, x)
            if yhat == y:
                correct += 1
        
        accuracy = correct / len(lenses_data)
        print(f"训练集上的准确率：{correct}/{len(lenses_data)} = {accuracy:.2%}")
        
    except FileNotFoundError:
        print("错误：找不到 lenses.txt 文件")
        print("请确保 lenses.txt 文件在当前目录下")
    except Exception as e:
        print(f"发生错误：{e}")
