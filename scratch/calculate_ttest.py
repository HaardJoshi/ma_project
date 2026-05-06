from scipy import stats
import numpy as np

m1_fold_aucs = [0.521, 0.538, 0.544, 0.531, 0.549]
m3_fold_aucs = [0.558, 0.571, 0.562, 0.567, 0.574]

t_stat, p_val = stats.ttest_rel(m3_fold_aucs, m1_fold_aucs)
diff = np.array(m3_fold_aucs) - np.array(m1_fold_aucs)
mean_diff = np.mean(diff)
df = len(diff) - 1

# 95% Confidence Interval
ci = stats.t.interval(0.95, df=df, loc=mean_diff, scale=stats.sem(diff))

print(f"Mean M1: {np.mean(m1_fold_aucs):.4f}")
print(f"Mean M3: {np.mean(m3_fold_aucs):.4f}")
print(f"Mean Diff: {mean_diff:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"df: {df}")
print(f"p-value: {p_val:.6f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
