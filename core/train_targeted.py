import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
import random
import seaborn as sns
from scipy import stats

# --- 字体设置 ---
# 尝试使用多种中文字体，确保在不同系统上都能正常显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
except Exception as e:
    print(f"警告: 无法设置中文字体, 可能导致中文显示异常。错误信息: {e}")

# --- 1. 数据加载与预处理 ---
file_path = './datasets/靶向_数据集.csv'
df = pd.read_csv(file_path)

# 直接删除不需要的列
df = df.drop(columns=[
    '患者姓名', '住院号','2年肿瘤控制(未控制=1，控制/删失=0)'
], errors='ignore')


# 定义新的目标变量和特征集
target = '1年肿瘤控制(未控制=1，控制/删失=0)'
y = df[target]


categorical_cols = [
    '性别（1=1，2=2）', '组织学状态（腺癌=1，鳞癌=2，小细胞癌=3，其他=4）',
    '原发肿瘤状态（稳定=1，进展=2，未知=3）', '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）',
    '颅外转移（是=1，否=2）', '颅外疾病控制情况（稳定=1，进展=2，未知=3，/=4）',
    '病灶部位（单病灶=1，多病灶=2）', '脑转移数目（1=1、2-4=2、5-10=3、>10=4）',
    'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）','神经功能基线差（差=1，好=0）','CK前神经功能分级',
    '靶向治疗（CK前三个月）','靶向治疗（CK后三个月）','转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）','实体瘤评价(CR=1,PR=2,SD=3,PD=4)',
    '3月肿瘤控制(未控制=1，控制=2)','6月肿瘤控制(未控制=1，控制=2)'
]

all_cols = list(df.columns)
numeric_cols = [col for col in all_cols if col not in categorical_cols + [target]]

# 对数值列进行类型转换和填充缺失值
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df[categorical_cols] = df[categorical_cols].astype('category')

# 划分训练/验证集
X = df[categorical_cols + numeric_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = pd.get_dummies(X_train, columns=categorical_cols)
X_test_processed = pd.get_dummies(X_test, columns=categorical_cols)
X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

print("--- 数据处理完成 ---")

# --- 打印最终送入模型的特征列表 ---
print("\n--- 最终送入模型的特征列表（共", len(X_train_processed.columns), "个） ---")
for feature in X_train_processed.columns:
    print(feature)
print("\n" + "="*50 + "\n")

# --- 2. 构建并训练 XGBoost 模型 ---
count_negative = (y_train == 0).sum()  # 负样本数量
count_positive = (y_train == 1).sum()  # 正样本数量
scale_pos_weight = count_negative / count_positive

print("\n--- 正在训练 XGBoost 模型 ---")
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)
model_xgb.fit(X_train_processed, y_train)

print("--- 模型训练完成 ---")
print("\n--- 模型在测试集上的表现 ---")
y_pred = model_xgb.predict(X_test_processed)
print(classification_report(y_test, y_pred))

# --- 3. SHAP 可解释性分析与图表保存 ---
print("\n--- 正在进行 SHAP 可解释性分析并保存图表 ---")
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_processed)

# 创建保存目录
output_dir = './report/targeted'
os.makedirs(output_dir, exist_ok=True)
print(f"--- 目标保存目录已确认: {os.path.abspath(output_dir)} ---")

# --- 3.1 绘制并保存 SHAP 特征重要性图 (柱状图) ---
print("--- 正在生成并保存 SHAP 特征重要性图 ---")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, plot_type="bar", show=False)
plt.title('SHAP 特征重要性排序图', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SHAP_feature_importance_bar.png'), dpi=300)
plt.close() # 关闭当前图表，释放内存

print(f"图表已成功保存至 {os.path.abspath(output_dir)} 目录下。")

# --- 3.2 绘制并保存 SHAP 依赖图（手动实现三色着色）---
print("\n--- 正在手动绘制 SHAP 依赖图（BED10× 靶向模式，三色）---")

# 定义特征名称和独热编码后的列名
bed_feature = 'BED10'
target_therapy_cols = {
    '无靶向治疗': '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_0',
    '辅助靶向治疗': '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_1',
    '全程靶向治疗': '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）_2'
}
colors = ['red', 'green', 'blue']
labels = ['无靶向治疗', '辅助靶向治疗', '全程靶向治疗']

# 确保所有特征都存在于数据集中
if bed_feature in X_test_processed.columns and all(col in X_test_processed.columns for col in target_therapy_cols.values()):
    try:
        plt.figure(figsize=(12, 9))

        # 遍历每种靶向模式，并分别绘制
        for i, (mode_name, col_name) in enumerate(target_therapy_cols.items()):
            # 筛选出当前模式下的样本
            mask = X_test_processed[col_name] == 1
            if mask.any():
                # 获取该模式下 BED10 特征的 SHAP 值和特征值
                shap_values_mode = shap_values[mask, X_test_processed.columns.get_loc(bed_feature)]
                feature_values_mode = X_test_processed[bed_feature][mask]

                # 绘制散点图
                plt.scatter(
                    feature_values_mode,
                    shap_values_mode,
                    label=labels[i],
                    color=colors[i],
                    alpha=0.6,
                    s=80
                )

        plt.title('SHAP 依赖图：BED10 对模型输出的贡献', fontsize=16)
        plt.xlabel('BED10 值', fontsize=12)
        plt.ylabel('BED10 特征的 SHAP 值', fontsize=12)
        plt.legend(title='靶向治疗模式')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'SHAP_dependence_bed10_by_target_3_colors.png'), dpi=300)
        plt.close()
        print("--- SHAP 依赖图已成功保存 ---")

    except Exception as e:
        print(f"警告: 无法生成 SHAP 依赖图。错误信息: {e}")
else:
    print(f"警告: 独热编码后的特征列未找到。请检查列名是否正确。")


# --- 4. 计算并输出 AUC 和 95% CI ---
print("\n--- 正在计算 AUC 指标 ---")

# 使用 predict_proba 获取测试集上每个样本属于正类的概率
y_pred_proba = model_xgb.predict_proba(X_test_processed)[:, 1]

# 计算 ROC-AUC 分数
roc_auc = roc_auc_score(y_test, y_pred_proba)

# --- 5. 计算 AUC 的 95% 置信区间 (CI) ---
print("\n--- 正在使用 Bootstrap 方法计算 AUC 的 95% 置信区间 (CI) ---")
n_bootstraps = 1000
rng_seed = 42
bootstrap_aucs = []

y_test_np = y_test.to_numpy()
y_pred_proba_np = y_pred_proba

for i in range(n_bootstraps):
    indices = np.random.choice(len(y_test_np), len(y_test_np), replace=True)

    if len(np.unique(y_test_np[indices])) < 2:
        continue

    auc_value = roc_auc_score(y_test_np[indices], y_pred_proba_np[indices])
    bootstrap_aucs.append(auc_value)

lower_ci = np.percentile(bootstrap_aucs, 2.5)
upper_ci = np.percentile(bootstrap_aucs, 97.5)

print(f"AUC 的 95% 置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]")

# --- 6. 绘制并保存 ROC 曲线，并在图例中显示 AUC 和 95% CI ---
print("\n--- 正在绘制并保存 ROC 曲线 (含 95% CI) ---")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc_curve = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'AUC = {roc_auc_curve:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)') # 修改为中文标签
plt.ylabel('真阳性率 (TPR)') # 修改为中文标签
plt.title('受试者工作特征曲线 (ROC)', fontsize=14) # 修改为中文标题
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ROC_curve_with_ci.png'), dpi=300)
plt.close()

print(f"ROC 曲线图（含CI）已成功保存至 {os.path.abspath(output_dir)} 目录下。")


# --- 7. 绘制并保存混淆矩阵图 ---
print("\n--- 正在绘制并保存混淆矩阵图 ---")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5,
            xticklabels=['未控制 (0)', '控制 (1)'],
            yticklabels=['未控制 (0)', '控制 (1)'])
plt.xlabel('预测值', fontsize=12)
plt.ylabel('真实值', fontsize=12)
plt.title('混淆矩阵', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

print(f"混淆矩阵图已成功保存至 {os.path.abspath(output_dir)} 目录下。")


# --- 8. 自动计算所有特征的 95% CI 并保存到txt ---
print("\n--- 正在计算所有特征的 95% 置信区间 ---")
output_txt_path = os.path.join(output_dir, 'all_features_confidence_intervals.txt')

with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write("模型所有特征的 95% 置信区间报告\n")
    f.write("="*40 + "\n")

    # 遍历所有特征
    for feature in X_test_processed.columns:
        feature_data = X_test_processed[feature]

        # 检查数据是否可用于计算（至少两个不重复的值）
        if len(feature_data.unique()) > 1:
            try:
                sample_mean = feature_data.mean()
                sample_size = len(feature_data)

                # 使用 t-分布计算置信区间
                confidence_interval = stats.t.interval(
                    confidence=0.95,
                    df=sample_size - 1,
                    loc=sample_mean,
                    scale=stats.sem(feature_data)
                )

                lower_ci, upper_ci = confidence_interval

                f.write(f"特征: {feature}\n")
                f.write(f"  均值: {sample_mean:.4f}\n")
                f.write(f"  95% 置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]\n")
                f.write("-" * 20 + "\n")
            except Exception as e:
                f.write(f"特征: {feature}\n")
                f.write(f"  警告: 无法计算此特征的置信区间。原因：{e}\n")
                f.write("-" * 20 + "\n")
        else:
            f.write(f"特征: {feature}\n")
            f.write("  警告: 数据变异性不足（所有值都相同），无法计算置信区间。\n")
            f.write("-" * 20 + "\n")

print(f"--- 所有特征的置信区间已成功保存至 {os.path.abspath(output_txt_path)} ---")