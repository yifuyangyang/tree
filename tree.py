import operator
from math import log
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心算法部分 (ID3)
# ==========================================

def calc_shannon_ent(dataset):
    """ 计算香农熵 """
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def split_dataset(dataset, axis, value):
    """ 
    划分数据集 
    axis: 按第几个特征划分
    value: 划分的值
    返回: 去掉了 axis 这一列的子数据集
    """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_to_split(dataset):
    """ 选择最好的特征进行划分 (信息增益) """
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        
        info_gain = base_entropy - new_entropy
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            
    # 如果没有特征能带来增益（或者所有特征遍历完），返回-1
    return best_feature

def majority_cnt(class_list):
    """ 多数表决 """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset, labels):
    """ 
    构建决策树 
    注意：labels 列表在函数内会被修改，所以调用时建议传副本 labels[:]
    """
    class_list = [example[-1] for example in dataset]
    
    # 1. 类别完全相同，停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 2. 遍历完所有特征，返回多数表决
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    best_feat = choose_best_feature_to_split(dataset)
    
    # 如果无法找到最优特征（增益为0），直接返回多数表决
    if best_feat == -1:
         return majority_cnt(class_list)

    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    
    # 删除已使用的特征标签
    del(labels[best_feat])
    
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    
    for value in unique_vals:
        sub_labels = labels[:] # 复制标签，防止递归污染
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
        
    return my_tree

def classify(input_tree, feature_labels, test_vec):
    """ 
    使用决策树分类 
    feature_labels: 完整的特征名称列表（用于定位特征索引）
    """
    first_str = next(iter(input_tree)) # 获取根节点特征名
    child_dict = input_tree[first_str]
    
    # 找到当前特征在 test_vec 中的索引
    try:
        feat_index = feature_labels.index(first_str)
    except ValueError:
        return "Unknown Feature"

    key = test_vec[feat_index]
    
    # 处理未知分支（例如训练集中没有这个特征值）
    if key not in child_dict:
        # 这里可以返回 "无法判断" 或者 默认类别
        # 为了准确率统计，我们标记为 Unknown
        return "Unknown"
        
    feat_value = child_dict[key]
    
    if isinstance(feat_value, dict):
        return classify(feat_value, feature_labels, test_vec)
    else:
        return feat_value

# ==========================================
# 2. 可视化部分 (Matplotlib)
# ==========================================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center", bbox=node_type, arrowprops=arrow_args,
                fontsize=10)

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30, fontsize=9)

def plot_tree_recursive(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    num_leafs = get_num_leafs(my_tree)
    first_str = next(iter(my_tree))
    # 计算节点位置
    center_pt = (x_off_y['x'] + (1.0 + float(num_leafs)) / 2.0 / total_w, x_off_y['y'])
    
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    
    second_dict = my_tree[first_str]
    x_off_y['y'] = x_off_y['y'] - 1.0 / total_d
    
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree_recursive(ax, second_dict[key], center_pt, str(key), total_w, total_d, x_off_y)
        else:
            x_off_y['x'] = x_off_y['x'] + 1.0 / total_w
            plot_node(ax, str(second_dict[key]), (x_off_y['x'], x_off_y['y']), center_pt, leaf_node)
            plot_mid_text(ax, (x_off_y['x'], x_off_y['y']), center_pt, str(key))
            
    x_off_y['y'] = x_off_y['y'] + 1.0 / total_d

def create_plot(my_tree):
    fig = plt.figure(1, facecolor='white', figsize=(12, 7))
    fig.clf()
    ax = plt.subplot(111, frameon=False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none') # 隐藏坐标轴
    
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x': -0.5 / total_w, 'y': 1.0}
    
    plot_tree_recursive(ax, my_tree, (0.5, 1.0), '', total_w, total_d, x_off_y)
    plt.show()

# ==========================================
# 3. 主程序 (数据加载与测试)
# ==========================================

if __name__ == "__main__":
    # >>>>> 请在这里修改你的文件路径 <<<<<
    file_path = r'C:\Users\E507\Desktop\tree\lenses.txt'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lenses_data = [line.strip().split() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {file_path}")
        lenses_data = []

    if lenses_data:
        print(f"成功读取 {len(lenses_data)} 条数据。")
        
        # 定义特征名称 (必须与 dataset 的列顺序对应)
        lenses_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
        
        # 备份 labels 供后续验证使用 (因为 create_tree 会修改传入的 labels)
        labels_for_test = lenses_labels[:] 

        # 1. 构建决策树
        # 传入 lenses_labels[:] 副本，保护原列表
        my_tree = create_tree(lenses_data, lenses_labels[:])
        
        print("\n生成的决策树字典：")
        print(my_tree)
        
        # 2. 绘制决策树
        print("\n正在绘图...")
        create_plot(my_tree)
        
        # 3. 计算训练集准确率
        correct_count = 0
        unknown_count = 0
        total_count = len(lenses_data)
        
        print("\n开始验证准确率...")
        for i, row in enumerate(lenses_data):
            # 提取特征和真实标签
            test_vec = row[:-1]
            true_label = row[-1]
            
            # 使用备份的标签列表进行预测
            pred_label = classify(my_tree, labels_for_test, test_vec)
            
            if pred_label == true_label:
                correct_count += 1
            elif pred_label == "Unknown":
                unknown_count += 1
                # print(f"样本 {i} 无法判断: {test_vec}") # 调试用
            else:
                pass 
                # print(f"样本 {i} 预测错误: 预测={pred_label}, 真实={true_label}") # 调试用
        
       
        
       
