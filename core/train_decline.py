import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import xgboost as xgb
import shap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay

# --- 字体设置 ---
# 尝试使用多种中文字体，确保在不同系统上都能正常显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
except Exception as e:
    print(f"警告: 无法设置中文字体, 可能导致中文显示异常。错误信息: {e}")

# --- 1. 数据加载与预处理 ---
file_path = './datasets/kps下降_数据集.csv'
df = pd.read_csv(file_path)

# 数据清理
df = df.drop(columns=['患者姓名', '住院号'], errors='ignore')

# 构建标签
df['KPS+神经功能级（3月）'] = df['KPS+神经功能级（3月）'].astype(str)
df['KPS_3月_num'] = pd.to_numeric(df['KPS+神经功能级（3月）'], errors='coerce')
df['KPS_diff3'] = -1
df.loc[df['KPS+神经功能级（3月）'] == '死亡', 'KPS_diff3'] = 0
mask = df['KPS_diff3'] == -1
diff = df.loc[mask, 'KPS'] - df.loc[mask, 'KPS_3月_num']
df.loc[mask & (diff < 0), 'KPS_diff3'] = 0
df.loc[mask & (diff == 0), 'KPS_diff3'] = 1
df.loc[mask & (diff > 0), 'KPS_diff3'] = 2
df['KPS_diff2'] = df['KPS_diff3'].apply(lambda x: 0 if x == 0 else 1)

# 特征工程
df['神经基线差'] = (df['CK前神经功能分级'] >= 2).astype(int)
df['BED10_神经基线差_交互'] = df['BED10'] * df['神经基线差']
df['BED10_KPS_交互'] = df['BED10'] * df['KPS']
df = df.drop(columns=['KPS+神经功能级（3月）', 'KPS_3月_num', 'KPS_diff3'])

# 定义目标变量和特征
target = 'KPS_diff2'
y = df[target]

categorical_cols = [
    '性别（1=1，2=2）', '组织学状态（腺癌=1，鳞癌=2，小细胞癌=3，其他=4）',
    '原发肿瘤状态（稳定=1，进展=2，未知=3）', '是否靶向治疗（全程靶向=2，辅助靶向=1，无靶向=0）',
    '颅外转移（是=1，否=2）', '颅外疾病控制情况（稳定=1，进展=2，未知=3，/=4）',
    '病灶部位（单病灶=1，多病灶=2）', '脑转移数目（1=1、2-4=2、5-10=3、>10=4）',
    'DS - GPA（0-1=1，1.5-2=2，2.5-3=3，3.5-4=4）','神经功能基线差（差=1，好=0）','CK前神经功能分级',
    '靶向治疗（CK前三个月）','靶向治疗（CK后三个月）','转移灶直径分组（≤2cm=1，2.01-3cm=2，>3cm=3）'
]

numeric_cols = [col for col in df.columns if col not in categorical_cols + [target]]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

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
print("--- 正在训练 XGBoost 模型 ---")
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model_xgb.fit(X_train_processed, y_train)

print("--- 模型训练完成 ---")

# --- 新增: 模型评估 ---
print("\n--- 正在生成模型分类报告 ---")
y_pred = model_xgb.predict(X_test_processed)
print(classification_report(y_test, y_pred, target_names=['KPS未下降', 'KPS下降']))
print("--- 分类报告生成完毕 ---")

# --- 3. 生成图表并保存 ---
print("\n--- 正在生成图表 ---")
output_dir = './report/decline'
os.makedirs(output_dir, exist_ok=True)

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_train_processed)

# --- 图1: 每个特征的重要性图 (SHAP Bar Plot) ---
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_train_processed, plot_type="bar", show=False)
plt.title('SHAP特征重要性')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_feature_importance_bar.png'))
plt.close()
print("SHAP特征重要性图已保存到 ./report/decline/shap_feature_importance_bar.png")

# --- 图2: BED10 SHAP 依赖图 ---
plt.figure(figsize=(10, 8))
shap.dependence_plot(
    ind='BED10',
    shap_values=shap_values,
    features=X_train_processed,
    interaction_index='神经基线差',
    show=False
)
plt.title('BED10 SHAP依赖图')
plt.xlabel('BED10剂量 (Gy)')
plt.ylabel('SHAP值 (量化对KPS下降的贡献)')
plt.legend(title='神经基线', labels=['好 (分级<2)', '差 (分级≥2)'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bed10_shap_dependence.png'))
plt.close()
print("BED10 SHAP依赖图已保存到 ./report/decline/bed10_shap_dependence.png")

# --- 图3: BED10 部分依赖图 (PDP) ---
features_to_plot = ['BED10']
fig, ax = plt.subplots(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(
    model_xgb,
    X_train_processed,
    features=features_to_plot,
    ax=ax,
    feature_names=X_train_processed.columns
)
ax.set_title('BED10 部分依赖图')
ax.set_xlabel('BED10剂量 (Gy)')
ax.set_ylabel('KPS下降预测概率')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bed10_partial_dependence.png'))
plt.close()
print("BED10 部分依赖图已保存到 ./report/decline/bed10_partial_dependence.png")


# --- 4. 计算并绘制 ROC 曲线及95% CI ---
print("\n--- 正在计算 AUC 并绘制 ROC 曲线 ---")
# 获取测试集上的预测概率
y_pred_proba = model_xgb.predict_proba(X_test_processed)[:, 1]

# 计算 ROC-AUC 分数
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 使用 Bootstrap 方法计算 95% 置信区间
n_bootstraps = 1000
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

print(f"模型在测试集上的 AUC: {roc_auc:.4f}")
print(f"AUC 的 95% 置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]")

# 绘制 ROC 曲线图
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc_curve = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'AUC = {roc_auc_curve:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('受试者工作特征曲线 (ROC)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ROC_curve_with_ci.png'), dpi=300)
plt.close()
print("ROC 曲线图已保存到 ./report/decline/ROC_curve_with_ci.png")


# --- 5. 绘制混淆矩阵图 ---
print("\n--- 正在绘制混淆矩阵图 ---")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5,
            xticklabels=['KPS未下降', 'KPS下降'],
            yticklabels=['KPS未下降', 'KPS下降'])
plt.xlabel('预测值', fontsize=12)
plt.ylabel('真实值', fontsize=12)
plt.title('混淆矩阵', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

print(f"混淆矩阵图已成功保存至 {os.path.abspath(output_dir)} 目录下。")

print("\n--- 所有任务完成 ---")
