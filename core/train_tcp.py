import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from pathlib import Path
from imblearn.combine import SMOTETomek
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据加载与预处理 ---
# 定义数据文件路径
file_path = './datasets/靶向_数据集.csv'
try:
    # 尝试加载数据
    df_raw = pd.read_csv(file_path)
except FileNotFoundError:
    # 如果文件未找到，打印错误信息并退出
    print(f"错误: 找不到文件 {file_path}。请确保文件路径正确。")
    exit()

# 定义目标变量和核心预测特征
targets = ['1年肿瘤控制(未控制=1，控制/删失=0)', '2年肿瘤控制(未控制=1，控制/删失=0)']
bed_feature = 'BED10'
stratify_col = '转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）'

# 定义模型参数
MAX_ITERATIONS = 100

# 确保所有必需的列都存在于数据集中
for col in [bed_feature, stratify_col] + targets:
    if col not in df_raw.columns:
        raise ValueError(f"数据集中未找到所需的特征列: {col}。请检查列名。")

# 将BED10列转换为数值类型，并将非数值数据填充为0
df_raw[bed_feature] = pd.to_numeric(df_raw[bed_feature], errors='coerce').fillna(0)

# 创建报告输出目录
output_dir = Path("report/tcp")
output_dir.mkdir(parents=True, exist_ok=True)

# 定义颜色和标记，用于区分不同分组
group_colors = {1: 'blue', 2: 'green', 3: 'red'}
group_markers = {1: 's', 2: 'o', 3: '^'}
group_names_map = {1: '<=2cm', 2: '2-3cm', 3: '>3cm'}
# 用于图表标题的映射
target_titles = {
    '1年肿瘤控制(未控制=1，控制/删失=0)': '1 Year Tumor Control Probability Curves',
    '2年肿瘤控制(未控制=1，控制/删失=0)': '2 Year Tumor Control Probability Curves'
}

# --- 2. 循环处理每个目标变量并保存分层分析结果 ---
# 遍历1年和2年肿瘤控制两个目标变量，并显示进度条
for target in tqdm(targets, desc="分析目标变量"):
    # 为每个目标变量创建一个独立的报告文件
    output_filename = output_dir / f"{target.split('(')[0]}_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        print(f"--- 正在对 '{target.split('(')[0]}' 进行分析 ---", file=f)

        # 复制原始数据以避免循环中的修改影响下一次循环
        df = df_raw.copy()

        # 确保目标变量只有0和1（将2替换为0）
        df[target] = df[target].replace({2: 0})

        # --- 2.1. 使用 Isolation Forest 移除异常值 ---
        # 孤立森林的输入是 BED10 和当前的目标变量
        print(f"\n--- 正在对 '{target.split('(')[0]}' 数据集使用孤立森林移除异常值 ---", file=f)

        # 创建一个副本以在异常值检测前填充缺失值
        df_impute = df.copy()
        df_impute[target] = df_impute[target].fillna(0) # 暂时填充缺失值以进行异常值检测

        iso_forest_input = df_impute[[bed_feature, target , stratify_col]].values

        # 初始化并训练孤立森林模型
        iso_forest = IsolationForest(
            n_estimators=500,
            contamination=0.2,
            max_features=2,
            random_state=42
        )
        iso_forest.fit(iso_forest_input)
        outlier_labels = iso_forest.predict(iso_forest_input)

        # 筛选出非异常值样本
        df_clean = df[outlier_labels == 1].copy()

        # 输出被移除的样本数量
        num_removed = len(df) - len(df_clean)
        print(f"原始数据量: {len(df)} 条", file=f)
        print(f"被孤立森林移除的异常样本数量: {num_removed} 条", file=f)
        print(f"移除异常值后数据量: {len(df_clean)} 条", file=f)
        print("-" * 50, file=f)

        # --- 2.2. 分层逻辑回归分析和数据收集 ---
        # 获取分层列的所有唯一值
        stratify_groups = sorted(df_clean[stratify_col].dropna().unique())
        p_values_list = []
        # 存储每个分组的拟合结果和原始数据，用于统一绘图
        fit_results_data = []

        for group_id in stratify_groups:
            group_name = group_names_map.get(group_id, str(group_id))
            print(f"\n--- 分析组别: 转移灶直径 {group_name} ---", file=f)

            # 筛选出当前组别的数据
            df_group = df_clean[df_clean[stratify_col] == group_id].copy()

            # 检查样本量和类别分布是否满足回归要求
            if len(df_group) < 10 or len(df_group[target].unique()) < 2:
                print(f"警告: 该组样本量不足 ({len(df_group)}个) 或类别不完整，无法进行逻辑回归。跳过此组。", file=f)
                continue

            # 准备自变量和因变量
            X = df_group[[bed_feature]]
            y_group = df_group[target]

            # --- 2.3. 使用 SMOTETomek 进行样本合成以平衡类别 ---
            # 检查类别是否不平衡，并应用 SMOTETomek
            if len(y_group.unique()) > 1 and len(y_group[y_group == 0]) != len(y_group[y_group == 1]):
                try:
                    # 实例化 SMOTETomek
                    smotetomek = SMOTETomek(random_state=42)
                    X_resampled, y_resampled = smotetomek.fit_resample(X, y_group)

                    X = X_resampled
                    y_group = y_resampled
                except ValueError as e:
                    print(f"  警告: 无法对该组应用 SMOTETomek。错误信息: {e}", file=f)

            # 为数据添加截距项
            X_sm = sm.add_constant(X)

            print(f"  当前组别样本量: {len(df_group)}", file=f)

            # 训练逻辑回归模型，使用MAX_ITERATIONS变量
            try:
                model = sm.Logit(y_group, X_sm)
                result = model.fit(disp=0, maxiter=MAX_ITERATIONS, method='cg')

                # 检查模型是否收敛以及BED10系数是否为0
                if not result.converged or bed_feature not in result.params or result.params[bed_feature] == 0:
                    print("警告: 模型未收敛或BED10系数为0，无法计算TCD₅₀。跳过此组。", file=f)
                    continue

                # 提取模型参数和协方差矩阵
                beta_0 = result.params['const']
                beta_1 = result.params[bed_feature]
                cov_matrix = result.cov_params()

                # --- 记录BED10系数的P值 ---
                p_value = result.pvalues[bed_feature]
                p_values_list.append(p_value)

                # 将每个分组的拟合结果和原始数据收集起来
                fit_results_data.append({
                    'group_id': group_id,
                    'group_name': group_name,
                    'beta_0': beta_0,
                    'beta_1': beta_1,
                    'cov_matrix': cov_matrix,
                    'df_group': df_group.copy()
                })

                # --- 计算 TCD₅₀ ---
                var_beta_0 = cov_matrix.loc['const', 'const']
                var_beta_1 = cov_matrix.loc[bed_feature, bed_feature]
                cov_beta_0_beta_1 = cov_matrix.loc['const', bed_feature]
                tcd_50 = -beta_0 / beta_1

                # --- 使用Delta方法计算95%置信区间 ---
                # 计算TCD₅₀的方差
                var_tcd_50 = (1 / beta_1**2) * var_beta_0 + \
                             (beta_0**2 / beta_1**4) * var_beta_1 - \
                             (2 * beta_0 / beta_1**3) * cov_beta_0_beta_1

                if var_tcd_50 <= 0:
                    print("警告: 计算的TCD₅₀方差为负或零，无法计算置信区间。", file=f)
                else:
                    # 计算标准误和置信区间
                    se_tcd_50 = np.sqrt(var_tcd_50)
                    ci_lower = tcd_50 - 1.96 * se_tcd_50
                    ci_upper = tcd_50 + 1.96 * se_tcd_50
                    print(f"成功计算TCD₅₀: {tcd_50:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])", file=f)

            except Exception as e:
                print(f"错误: 无法对该组拟合逻辑回归模型。错误信息: {e}", file=f)
                continue

        # --- 2.4. 统一绘制拟合曲线图 ---
        if fit_results_data:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(12, 8))

            # 创建一个用于绘制平滑曲线的BED10范围，覆盖所有分组的数据范围
            x_min = min(item['df_group'][bed_feature].min() for item in fit_results_data)
            x_max = max(item['df_group'][bed_feature].max() for item in fit_results_data)
            x_range = np.linspace(x_min - 5, x_max + 5, 500)

            # 绘制所有分组的曲线和置信区间
            for data in fit_results_data:
                beta_0 = data['beta_0']
                beta_1 = data['beta_1']
                cov_matrix = data['cov_matrix']
                group_id = data['group_id']
                group_name = data['group_name']

                # 计算预测概率和置信区间
                pred_prob = 1 / (1 + np.exp(-(beta_0 + beta_1 * x_range)))

                var_beta_0 = cov_matrix.loc['const', 'const']
                var_beta_1 = cov_matrix.loc[bed_feature, bed_feature]
                cov_beta_0_beta_1 = cov_matrix.loc['const', bed_feature]

                g_prime_beta0 = pred_prob * (1 - pred_prob)
                g_prime_beta1 = pred_prob * (1 - pred_prob) * x_range
                var_pred = g_prime_beta0**2 * var_beta_0 + \
                           g_prime_beta1**2 * var_beta_1 + \
                           2 * g_prime_beta0 * g_prime_beta1 * cov_beta_0_beta_1
                var_pred[var_pred < 0] = 0
                se_pred = np.sqrt(var_pred)
                ci_lower = pred_prob - 1.96 * se_pred
                ci_upper = pred_prob + 1.96 * se_pred

                color = group_colors.get(group_id, 'black')

                # 绘制曲线和置信区间
                plt.plot(x_range, pred_prob, color=color, linewidth=2, label=f'Fit Curve: {group_name}')
                plt.fill_between(x_range, ci_lower, ci_upper, color=color, alpha=0.1)

            # 绘制所有分组的原始数据点
            for data in fit_results_data:
                group_id = data['group_id']
                group_name = data['group_name']
                df_group = data['df_group']
                color = group_colors.get(group_id, 'black')
                marker = group_markers.get(group_id, 's')

                sns.scatterplot(
                    x=df_group[bed_feature],
                    y=df_group[target],
                    marker=marker,
                    color=color,
                    s=50,
                    alpha=0.6,
                    label=f'Observed Data: {group_name}'
                )

            # 设置图表标题和标签
            plt.title(target_titles[target], fontsize=18)
            plt.xlabel('BED10', fontsize=14)
            plt.ylabel('Probability of Control', fontsize=14)
            plt.legend(title='Groups', loc='lower right')
            plt.ylim(-0.05, 1.05)
            plt.xlim(x_range.min(), x_range.max())

            # 保存图像
            plot_filename = output_dir / f"{target.split('(')[0]}_curves.png"
            plt.savefig(plot_filename)
            plt.close() # 关闭图形，防止内存泄漏

        else:
            print("  警告: 没有足够的数据生成统一的拟合曲线图。", file=f)

        # --- 输出平均P值 ---
        if p_values_list:
            average_p = np.mean(p_values_list)
            print(f"\nP值: P = {average_p:.4f}", file=f)

    print(f"\n'{target.split('(')[0]}' 的分析已完成。结果已保存到 {output_filename}")

print("\n所有分析均已完成，请检查 'report/tcp' 目录下的报告文件。")