from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

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
        # 根据香农熵公式累加：
        shannon_ent -= prob*log(prob, 2)
    # 5. 返回计算得到的熵值
    return shannon_ent


# def create_dataSet():
    
#     dataset = [[1, 1, 'yes'],
#                [1.1, 'yes'],
#                [1, 0, 'no'],
#                [0, 1, 'no'],
#                [0, 1, 'no']]
#     labels = ['no suerfacing', 'flippers']
#     return dataset, labels


# dataset, labels = create_dataSet()
# print(cal_shannon_ent(dataset))


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
            # 构建一个“去掉该特征”的新样本
            reduced_feat_vec = feat_vec[:axis]    # 取前面部分
            reduced_feat_vec.extend(feat_vec[axis+1:])  # 取后面部分拼接起来
            # 把这个新样本加入到子数据集中
            ret_dataset.append(reduced_feat_vec)
      # 返回划分后的数据集
    return ret_dataset


# 示例数据集：最后一列是标签
#

# # 按第0列的值为1来划分
# result = split_dataset(dataset_test, 0, 1)
# print(result)


def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征索引，作为本轮划分的最优特征。

    参数：
        dataset: 数据集（二维列表，每行一条样本，最后一列是标签）
    返回：
        best_feature: 最优特征的索引位置
    """
    # 1. 计算特征总数（最后一列是标签，不算特征）
    num_features = len(dataset[0])-1
    # 2. 计算原始数据集的熵（未划分前的不确定性）
    base_entropy = cal_shannon_ent(dataset)
    # 3. 初始化“最大信息增益”和“最佳特征”
    best_info_gain = 0.0
    best_feature = 1
    # 4. 遍历每一个特征，计算它的信息增益
    for i in range(num_features):
        # 4.1 提取出该特征所有样本的取值列表
        feat_list = [example[i] for example in dataset]
        #这是一个列表推导式的写法
        #等价于:
        #feat_list = []
        #for example in dataset:
        #    feat_list.append(example[i])
        # 4.2 获取该特征的所有唯一取值,转换为set集合，自动去重
        unique_val = set(feat_list)
        # 4.3 计算该特征划分后的“加权平均熵”
        new_entropy = 0.0
        for value in unique_val:
            # 按照该特征的某个取值划分数据集
            sub_dataset = split_dataset(dataset, i, value)
             # 计算该子集占整个数据集的比例
            prob = len(sub_dataset)/float(len(dataset))
            # 累加加权熵（概率 * 子集熵）
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        # 4.4 计算该特征的信息增益
        info_gain = base_entropy-new_entropy
        # 4.5 如果当前特征信息增益更大，就更新最优特征
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    # 5. 返回信息增益最大的特征索引
    return best_feature

#print(choose_best_feature_split(loan_data))

def majority_cnt(class_list):
    """
    功能：统计 class_list 中各类别出现的次数，并按出现次数从多到少排序返回。
    参数：
        class_list: 列表，例如 ['yes', 'no', 'yes', 'yes', 'no']
    返回：
        一个按类别出现次数从多到少排列的列表，例如：
        [('yes', 3), ('no', 2)]
    """
     # 1. 定义一个空字典，用于存放每个类别及其计数
    class_count={}
    # 2. 遍历类别列表，对每个类别进行计数
    for vote in class_list:
        # 如果该类别还未在字典中出现，先初始化计数为0
        if vote not in class_count.keys():class_count[vote]=0
        # 累加该类别的出现次数
        class_count[vote]+=1
    # 3. 将字典的键值对（类别, 次数）转为列表，并按次数进行降序排序
    # operator.itemgetter(1) 表示按照元组中第2个元素（计数）排序
    # dict.items() => [('yes',3), ('no',2)]
    # 按出现次数排序
     # 降序排列
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count

def creat_tree(dataset,labels):
    # 取出数据集中每条样本的“标签列”（通常是最后一列）
    class_list=[example[-1] for example in dataset]
    # 递归出口①：若所有样本同类，直接返回该类
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
    # 递归出口②：若没有可用特征（只剩标签列），返回多数类
    # dataset[0] 的长度 = 特征数 + 1（标签列）
    if len(dataset[0])==1:
        return majority_cnt(class_list)
    # 选择“最优划分特征”的下标
    best_feat=choose_best_feature_split(dataset)
     # 取出该特征对应的名称（可读性用）
    best_feat_label=labels[best_feat]
    # 构建当前节点
    my_tree={best_feat_label:{}}
    del(labels[best_feat])
     # 取出该特征在所有样本上的取值列表
    feat_values=[example[best_feat] for example in dataset]
    # 去重：该特征有哪些不同的取值
    unique_vals=set(feat_values)
    # 对该特征的每个取值，分别递归构建子树
    for value in unique_vals:
        sub_labels=labels[:]   # 拷贝一份标签名列表给子递归使用
        # 把当前特征=某取值的样本切分出来
        my_tree[best_feat_label][value]=creat_tree(split_dataset(dataset,best_feat,value),sub_labels)
    return my_tree

# my_data,labels=create_dataSet()
# my_tree=creat_tree(my_data,labels)


# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


#decision_node：定义“决策节点”的外观样式。
#boxstyle="sawtooth" 表示锯齿边框，常用于显示决策节点；
#fc='0.8'（facecolor）填充颜色为灰白色（0.8 表示灰度级）。
decision_node=dict(boxstyle="sawtooth",fc='0.8')

#leaf_node：定义“叶节点”的样式。
#boxstyle="round4" 表示圆角矩形边框；
#fc='0.8' 同样灰白填充。
leaf_node=dict(boxstyle="round4",fc='0.8')

#arrow_args：定义箭头样式。
#arrowstyle="<-" 表示箭头方向从子节点指向父节点。
arrow_args=dict(arrowstyle="<-")

# #node_txt：节点文字（显示在框中的文字，如“决策节点”、“叶节点”）。
# #center_pt：节点中心位置（子节点的位置）。
# #parent_pt：父节点位置，用于绘制箭头的起点。
# #node_type：节点样式（decision_node 或 leaf_node）。
# def plot_node(node_txt,center_pt,parent_pt,node_type): 
#     #annotate()：用于在图中添加带箭头的注释（文字+箭头）。
#     #xy=parent_pt：箭头起点（父节点位置）。
#     #xytext=center_pt：箭头终点+文字显示位置（子节点位置）。
#     #xycoords='axes fraction'：说明坐标用的是“轴的比例坐标”，即 (0,0) 是左下角，(1,1) 是右上角；
#     #bbox=node_type：节点边框样式；
#     #arrowprops=arrow_args：箭头样式；
#     #va='center'，ha='center'：文字居中对齐。
#     create_plot.ax1.annotate(node_txt,xy=parent_pt,xycoords='axes fraction',
#                              xytext=center_pt,textcoords='axes fraction',
#                              va='center',ha='center',bbox=node_type,arrowprops=arrow_args)
    

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')
    
def create_plot():
    fig=plt.figure(1,facecolor='white')  ## 新建一张图，背景白色
    fig.clf()                             # 清空之前的内容（防止重叠）
    create_plot.ax1=plt.subplot(111,frameon=False) # 创建一个子图，不显示坐标轴边框
    plot_node('决策节点',(0.5,0.1),(0.1,0.5),decision_node) # 画一个决策节点,节点位置 (0.5, 0.1)，箭头从 (0.1, 0.5) 指向节点；
    plot_node('叶节点',(0.8,0.1),(0.3,0.8),leaf_node) # 画一个叶节点,节点位置 (0.8, 0.1)，箭头从 (0.3, 0.8) 指向节点。
    plt.show()                                        # 显示图像

def get_num_leafs(my_tree):
    # my_tree 形如 {'特征A': {value1: 'yes', value2: {'特征B': {...}}}}
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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}

    plot_tree(ax, my_tree, parent_pt=(0.5, 1.0), node_txt='',
              total_w=total_w, total_d=total_d, x_off_y=x_off_y)

    plt.tight_layout()
    plt.show()

# ========== 运行：建树 + 绘图 ==========
# 示例数据集：天气与打球 (Play Tennis)
# weather_data = [
#     ['Sunny', 'Hot', 'High', False, 'No'],
#     ['Sunny', 'Hot', 'High', True, 'No'],
#     ['Overcast', 'Hot', 'High', False, 'Yes'],
#     ['Rain', 'Mild', 'High', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', True, 'No'],
#     ['Overcast', 'Cool', 'Normal', True, 'Yes'],
#     ['Sunny', 'Mild', 'High', False, 'No'],
#     ['Sunny', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'Normal', False, 'Yes'],
#     ['Sunny', 'Mild', 'Normal', True, 'Yes'],
#     ['Overcast', 'Mild', 'High', True, 'Yes'],
#     ['Overcast', 'Hot', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'High', True, 'No']
# ]



lenspath = (r'C:\Users\SOMEO\Desktop\新建文件夹\tree\lenses.txt')

def load_data(filepath):
    data=[]
    fr=open(filepath)
    for line in fr:
        line=line.strip().split('\t')
        data.append(line)
    return data
labels=['年龄','屈光','散光','泪液分泌']
dataset=load_data(lenspath)
tree=creat_tree(dataset,labels[:])
create_plot(tree)

# 决策树分类函数
def classify(inputTree, featLabels, testVec):
    """
    使用决策树进行分类
    """
    # 获取决策树的第一个键（根节点特征）
    firstStr = list(inputTree.keys())[0]
    # 获取子节点
    secondDict = inputTree[firstStr]
    # 查找特征在标签列表中的索引位置
    featIndex = featLabels.index(firstStr)
    
    # 遍历子节点
    for key in secondDict.keys():
        # 如果测试数据的特征值等于当前节点的键值
        if testVec[featIndex] == key:
            # 如果子节点仍然是字典（非叶子节点），递归调用
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: 
                # 到达叶子节点，返回分类结果
                classLabel = secondDict[key]
            return classLabel
    
    # 如果没有匹配的路径，返回默认值
    return "未知分类"

def single_case_test():
    """
    单病例测试功能
    """
    print("\n" + "="*50)
    print("决策树分类测试界面")
    print("="*50)
    print("请输入患者信息（格式：年龄 屈光 散光 泪液分泌）")
    print("例如" \
    "")
    print("输入 'quit' 或 '退出' 可以结束程序")
    print("-"*50)
    
    while True:
        try:
            print("\n请输入患者数据：")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', '退出', 'exit', 'q']:
                print("程序结束，再见！")
                break
                
            if not user_input:
                continue
                
            # 分割输入数据
            input_data = user_input.split()
            
            # 检查数据格式
            if len(input_data) != 4:
                print("错误：请输入4个特征值，用空格分隔")
                print("格式：年龄 屈光 散光 泪液分泌")
                print("例如：young myope no normal")
                continue
            
            # 提取特征值
            age, prescription, astigmatic, tear_rate = input_data
            
            # 验证特征值格式
            valid_ages = ['young', 'pre', 'presbyopic']
            valid_prescriptions = ['myope', 'hyper']
            valid_astigmatic = ['no', 'yes']
            valid_tear_rates = ['reduced', 'normal']
            
            if age not in valid_ages:
                print(f"错误：年龄必须是 {valid_ages} 中的一个")
                continue
            if prescription not in valid_prescriptions:
                print(f"错误：屈光必须是 {valid_prescriptions} 中的一个")
                continue
            if astigmatic not in valid_astigmatic:
                print(f"错误：散光必须是 {valid_astigmatic} 中的一个")
                continue
            if tear_rate not in valid_tear_rates:
                print(f"错误：泪液分泌必须是 {valid_tear_rates} 中的一个")
                continue
            
            # 构建测试数据
            test_data = [age, prescription, astigmatic, tear_rate]
            
            # 进行分类
            print("正在分类...")
            result = classify(tree, labels, test_data)
            
            # 显示结果
            print("\n" + "="*40)
            print("分类结果")
            print("="*40)
            print(f"输入数据: {test_data}")
            print(f"推荐镜片类型: {result}")
            
            # 解释结果
            if result == "no lenses":
                print("建议: 不适合佩戴隐形眼镜")
            elif result == "soft":
                print("建议: 推荐软性隐形眼镜")
            elif result == "hard":
                print("建议: 推荐硬性隐形眼镜")
            else:
                print("建议: 未知类型，请咨询专业医生")
            print("="*40)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"处理错误: {e}")
            print("请重新输入正确的数据格式")

def calculate_accuracy(tree, feature_labels, dataset):
    """
    计算决策树在训练集上的准确率
    """
    correct_predictions = 0
    total_samples = len(dataset)
    
    print("\n" + "="*60)
    print("训练集准确率计算")
    print("="*60)
    
    for i, sample in enumerate(dataset, 1):
        # 分离特征和真实标签（假设最后一列是标签）
        features = sample[:-1]  # 前n-1列是特征
        true_label = sample[-1]  # 最后一列是真实标签
        
        # 使用决策树进行预测
        predicted_label = classify(tree, feature_labels, features)
        
        # 检查预测是否正确
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
        
        # 打印每个样本的预测结果
        status = "✓" if is_correct else "✗"
        print(f"样本{i}: {features} -> 预测: {predicted_label:8} | 真实: {true_label:8} {status}")
    
    # 计算准确率
    accuracy = correct_predictions / total_samples * 100
    
    print("-"*60)
    print(f"总计: {total_samples} 个样本")
    print(f"正确: {correct_predictions} 个")
    print(f"错误: {total_samples - correct_predictions} 个")
    print(f"准确率: {accuracy:.2f}%")
    print("="*60)
    
    return accuracy

# 主程序
if __name__ == "__main__":
    print("决策树训练完成！")
    print("特征标签:", labels)
    accuracy = calculate_accuracy(tree, labels, dataset)
    print("准确率为：",accuracy)
    # 开始单病例测试
    single_case_test()