import pandas as pd
import numpy as np
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings

# 忽略不收敛警告，因为我们知道问题所在并正在尝试解决
warnings.filterwarnings("ignore", category=UserWarning)

# 设置中文字符的宽度（一个中文算作两个英文字符）
def get_padded_text(text, width):
    """
    根据字符的显示宽度进行填充，实现固定宽度的对齐
    一个中文字符被视为两个英文字符宽
    """
    display_width = sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)
    padding_needed = width - display_width
    if padding_needed > 0:
        return text + ' ' * padding_needed
    return text

# --- 1. 数据加载与预处理 ---
file_path = './datasets/靶向_数据集.csv'
try:
    df_raw = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}。请确保文件路径正确。")
    exit()

# 定义目标变量列表
targets = [
    '1年肿瘤控制(未控制=1，控制/删失=0)',
    '2年肿瘤控制(未控制=1，控制/删失=0)'
]

# 定义所有自变量
features = [
    'BED10',
    'GTV',
    '转移灶最大直径（cm）',
    '脑转移数目（1=1、2-4=2、5-10=3、>10=4）',
    '组织学状态（腺癌=1，鳞癌=2，小细胞癌=3，其他=4）',
    '原发肿瘤状态（稳定=1，进展=2，未知=3）',
    '颅外转移（是=1，否=2）',
    'KPS',
    'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）',
    'RPA',
    '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）',
    '病灶部位（单病灶=1，多病灶=2）'
]

# 确保所有必需的列都存在
required_cols = features + targets + ['转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）']
for col in required_cols:
    if col not in df_raw.columns:
        raise ValueError(f"数据集中未找到所需的特征列: {col}。请检查列名。")

# 数据类型转换
df_raw['BED10'] = pd.to_numeric(df_raw['BED10'], errors='coerce').fillna(0)
for target in targets:
    df_raw[target] = df_raw[target].replace({2: 0})

# 创建报告输出目录
output_dir = Path("report/tcp_multivariate")
output_dir.mkdir(parents=True, exist_ok=True)

# 循环处理每个目标变量
for target in targets:
    print(f"--- 正在进行 '{target.split('(')[0]}' 的多变量分析 ---")

    # --- 2. 数据清洗与特征工程 ---
    df_clean = df_raw.copy()

    categorical_cols = [
        '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）',
        '组织学状态（腺癌=1，鳞癌=2，小细胞癌=3，其他=4）',
        '原发肿瘤状态（稳定=1，进展=2，未知=3）',
        '颅外转移（是=1，否=2）',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）',
        '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）',
        '病灶部位（单病灶=1，多病灶=2）'
    ]
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')

    df_processed = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=False, dtype=int)

    X_cols = [
        'BED10', 'GTV', '转移灶最大直径（cm）', 'KPS', 'RPA',
        '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_1',
        '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_2',
        '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_3',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_1',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_2',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_3',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_4',
        '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_1',
        '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_2',
        '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_3',
        '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_4',
        '原发肿瘤状态（稳定=1，进展=2，未知=3）_1',
        '原发肿瘤状态（稳定=1，进展=2，未知=3）_2',
        '原发肿瘤状态（稳定=1，进展=2，未知=3）_3',
        '颅外转移（是=1，否=2）_1',
        '颅外转移（是=1，否=2）_2',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_1',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_2',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_3',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_4',
        '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_0',
        '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_1',
        '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_2',
        '病灶部位（单病灶=1，多病灶=2）_1',
        '病灶部位（单病灶=1，多病灶=2）_2'
    ]

    X_final_cols = [col for col in X_cols if col in df_processed.columns]
    X = df_processed[X_final_cols]
    y = df_processed[target]

    df_model = pd.concat([X, y], axis=1).dropna()
    X_model = df_model.drop(columns=[target])
    y_model = df_model[target]

    if len(y_model) < 10 or len(y_model.unique()) < 2:
        print("警告: 样本量不足或类别不完整，无法进行逻辑回归。")
        continue

    # --- 3. 针对特定特征的子类别进行SMOTE过采样 ---
    # 定义需要分层平衡的分类特征（独热编码后的列名）
    features_to_stratify = [
        '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）',
        '脑转移数目（1=1、2-4=2、5-10=3、>10=4）',
        '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）',
        '原发肿瘤状态（稳定=1，进展=2，未知=3）',
        '颅外转移（是=1，否=2）',
        'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）',
        '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）',
        '病灶部位（单病灶=1，多病灶=2）'
    ]

    smote_sampler = SMOTE(random_state=42)
    df_balanced_parts = []

    print(f"\n--- 开始分层过采样... ---")

    # 将处理后的数据集作为基础
    df_temp = pd.concat([X_model, y_model], axis=1)

    # 遍历每个分类特征
    for feature_name in features_to_stratify:
        # 获取该特征的独热编码后的所有列名
        one_hot_cols = [col for col in df_temp.columns if col.startswith(feature_name)]
        if not one_hot_cols:
            continue

        # 遍历每个子类别
        for one_hot_col in one_hot_cols:
            print(f"检查特征 '{one_hot_col.split('_', 1)[1]}'...")

            # 筛选出属于当前子类别的样本
            df_subset = df_temp[df_temp[one_hot_col] == 1].copy()

            if len(df_subset) < 10 or len(df_subset[target].unique()) < 2:
                print(f"  - 样本量过少或类别不完整，跳过。")
                df_balanced_parts.append(df_subset)
                continue

            X_subset = df_subset.drop(columns=[target])
            y_subset = df_subset[target]

            value_counts = y_subset.value_counts()
            is_imbalanced = value_counts.max() / value_counts.min() > 1.5 # 设定一个不平衡阈值

            if is_imbalanced:
                print(f"  - 发现不平衡。原始分布: \n{value_counts}")
                try:
                    X_res, y_res = smote_sampler.fit_resample(X_subset, y_subset)
                    df_res = pd.concat([X_res, y_res], axis=1)
                    df_balanced_parts.append(df_res)
                    print(f"  - SMOTE后分布: \n{y_res.value_counts()}")
                except ValueError as e:
                    print(f"  - 无法对该子集应用SMOTE。错误信息: {e}")
                    df_balanced_parts.append(df_subset)
            else:
                print(f"  - 类别已平衡，不进行过采样。")
                df_balanced_parts.append(df_subset)

    # 合并所有处理过的子集，并去除重复的样本
    df_balanced = pd.concat(df_balanced_parts).drop_duplicates().reset_index(drop=True)

    # 最终用于建模的数据集
    X_model_final = df_balanced.drop(columns=[target])
    y_model_final = df_balanced[target]

    print(f"\n--- 最终数据集类别分布: \n{y_model_final.value_counts()}")

    # --- 4. 拟合多变量逻辑回归模型 ---
    try:
        print(f"\n--- 正在拟合多变量Logistic回归模型 ---")
        model = sm.Logit(y_model_final, X_model_final)
        result = model.fit(disp=0, maxiter=200, method='cg')

        params = result.params
        conf_int = result.conf_int()
        pvalues = result.pvalues

        or_values = np.exp(params)
        ci_lower = np.exp(conf_int[0])
        ci_upper = np.exp(conf_int[1])

        results_df = pd.DataFrame({
            '变量': or_values.index,
            '调整后OR': or_values.values,
            '95% CI Lower': ci_lower.values,
            '95% CI Upper': ci_upper.values,
            'P值': pvalues.values
        })

        results_df['调整后OR'] = results_df['调整后OR'].round(2)
        results_df['95% CI'] = results_df.apply(
            lambda row: f"({row['95% CI Lower']:.2f}, {row['95% CI Upper']:.2f})", axis=1
        )
        results_df['P值'] = results_df['P值'].apply(
            lambda x: f"<0.001" if x < 0.001 else f"{x:.3f}"
        )

        results_df = results_df.drop(columns=['95% CI Lower', '95% CI Upper'])

        rename_map = {
            'BED10': 'BED10 (每增加1Gy)',
            'GTV': 'GTV',
            '转移灶最大直径（cm）': '转移灶最大直径（cm）',
            'KPS': 'KPS',
            'RPA': 'RPA',
            '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_1': '肿瘤大小: ≤2cm',
            '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_2': '肿瘤大小: 2-3 cm',
            '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）_3': '肿瘤大小: >3 cm',
            '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_1': '脑转移数目: 1',
            '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_2': '脑转移数目: 2-4',
            '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_3': '脑转移数目: 5-10',
            '脑转移数目（1=1、2-4=2、5-10=3、>10=4）_4': '脑转移数目: >10',
            '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_1': '组织学状态: 腺癌',
            '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_2': '组织学状态: 鳞癌',
            '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_3': '组织学状态: 小细胞癌',
            '组织学状态（腺癌=1，鳞癌=2、小细胞癌=3，其他=4）_4': '组织学状态: 其他',
            '原发肿瘤状态（稳定=1，进展=2，未知=3）_1': '原发肿瘤状态: 稳定',
            '原发肿瘤状态（稳定=1，进展=2，未知=3）_2': '原发肿瘤状态: 进展',
            '原发肿瘤状态（稳定=1，进展=2，未知=3）_3': '原发肿瘤状态: 未知',
            '颅外转移（是=1，否=2）_1': '颅外转移: 是',
            '颅外转移（是=1，否=2）_2': '颅外转移: 否',
            'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_1': 'DS - GPA: 0-1',
            'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_2': 'DS - GPA: 1.5-2',
            'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_3': 'DS - GPA: 2.5-3',
            'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）_4': 'DS - GPA: 3.5-4',
            '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_0': '是否靶向治疗: 无靶向',
            '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_1': '是否靶向治疗: 辅助靶向',
            '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_2': '是否靶向治疗: 全程靶向',
            '病灶部位（单病灶=1，多病灶=2）_1': '病灶部位: 单病灶',
            '病灶部位（单病灶=1，多病灶=2）_2': '病灶部位: 多病灶'
        }
        results_df['变量'] = results_df['变量'].replace(rename_map)

        # --- 5. 手动格式化输出到txt以确保完美对齐 ---
        file_base_name = target.split('(')[0].replace(' ', '_').replace('年肿瘤控制', 'yr').lower()
        output_filename = output_dir / f"tcp_{file_base_name}_multivariate.txt"

        # 设置固定的列宽，您可以根据需要调整这些值
        column_widths = {
            '变量': 30,
            '调整后OR': 15,
            '95% CI': 25,
            'P值': 10
        }

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"表: {target.split('(')[0]} 的多变量Logistic回归分析 (调整后OR值)\n")
            f.write("-" * 80 + "\n")

            # 写入表头
            header_line = (
                    get_padded_text('变量', column_widths['变量']) +
                    get_padded_text('调整后OR', column_widths['调整后OR']) +
                    get_padded_text('95% CI', column_widths['95% CI']) +
                    get_padded_text('P值', column_widths['P值'])
            )
            f.write(header_line + "\n")
            f.write("-" * sum(column_widths.values()) + "\n")

            # 写入每一行数据
            for _, row in results_df.iterrows():
                row_line = (
                        get_padded_text(row['变量'], column_widths['变量']) +
                        get_padded_text(str(row['调整后OR']), column_widths['调整后OR']) +
                        get_padded_text(row['95% CI'], column_widths['95% CI']) +
                        get_padded_text(row['P值'], column_widths['P值'])
                )
                f.write(row_line + "\n")

            f.write("\n\n")

        print(f"分析完成。结果已保存到 {output_filename}")

    except Exception as e:
        print(f"错误: 无法拟合多变量逻辑回归模型。错误信息: {e}")
        continue

print("\n所有分析均已完成，请检查 'report/tcp_multivariate' 目录下的报告文件。")