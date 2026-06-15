import matplotlib.pyplot as plt

# 假设以下变量已定义：

# ResMLP 的测试集结果（需从 evaluate 或训练日志中获取）
resmlp_test_accuracy = test_accuracies[-1]  # 最终测试准确率
resmlp_test_precision = 0.85  # 示例值
resmlp_test_recall = 0.83     # 示例值
resmlp_test_f1 = 0.84         # 示例值


# GBT 的最终测试集指标（从 plot_gbt_training_progress_multiclass 中提取）
gbt_test_accuracy = test_scores[-1]
gbt_test_precision = test_precisions[-1]
gbt_test_recall = test_recalls[-1]
gbt_test_f1 = test_f1s[-1]

# 绘制柱状图
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
resmlp_values = [resmlp_test_accuracy, resmlp_test_precision, resmlp_test_recall, resmlp_test_f1]
gbt_values = [gbt_test_accuracy, gbt_test_precision, gbt_test_recall, gbt_test_f1]

x_indexes = range(len(metrics))

bar_width = 0.35
plt.figure(figsize=(10, 6))

# ResMLP 柱状图
bars1 = plt.bar(x_indexes, resmlp_values, width=bar_width, label='ResMLP', color='skyblue')

# GBT 柱状图
bars2 = plt.bar([i + bar_width for i in x_indexes], gbt_values, width=bar_width, label='GBT', color='salmon')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# 设置图表信息
plt.xticks([i + bar_width/2 for i in x_indexes], metrics)
plt.ylabel('Score')
plt.title('Comparison of ResMLP and GBT on Test Set Metrics')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
