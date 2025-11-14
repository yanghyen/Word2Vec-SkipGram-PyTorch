import numpy as np
from scipy import stats
import pandas as pd
import os

# ===== 평가 결과 데이터 =====
evaluation_data = {
    'hs_window-2_sub-True': {
        'WordSim-353': [0.6783, 0.6747, 0.6649],
        'SimLex-999': [0.3019, 0.2940, 0.2894],
        'Google Analogy': [0.3078, 0.3116, 0.3162]
    },
    'hs_window-5_sub-True': {
        'WordSim-353': [0.6965, 0.6932, 0.7074],
        'SimLex-999': [0.2869, 0.2782, 0.2902],
        'Google Analogy': [0.3429, 0.3449, 0.3454]
    }
}

# ===== 학습 메트릭 데이터 =====
training_metrics = {
    'hs_window-2_sub-True': {
        'loss': [0.5986, 0.5987, 0.5986],
        'duration': [22735.11, 22824.95, 22856.37],
        'gpu_memory': [3.53, 3.53, 3.53]
    },
    'hs_window-5_sub-True': {
        'loss': [0.6145, 0.6145, 0.6145],
        'duration': [45841.53, 45761.36, 45787.97],
        'gpu_memory': [3.53, 3.53, 3.53]
    },
    'ns_window-2_sub-True': {
        'loss': [2.0327, 2.0307, 2.0326],
        'duration': [17539.07, 38295.25, 17553.21],
        'gpu_memory': [2.35, 3.53, 2.35]
    },
    'ns_window-5_sub-False': {
        'loss': [2.1973, 2.2041, 2.2011],
        'duration': [37255.78, 37335.02, 37270.55],
        'gpu_memory': [2.35, 2.35, 2.35]
    }
}

def calculate_stats(values):
    """평균, 표준편차 계산"""
    if len(values) < 2:
        return {
            'mean': values[0] if values else 0,
            'std': 0,
            'n': len(values)
        }
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    return {
        'mean': mean,
        'std': std,
        'n': len(values)
    }

# eval 폴더 생성
os.makedirs('eval', exist_ok=True)

# ===== CSV 파일 생성 =====

# 1. 평가 결과 통계 CSV
eval_results = []
for model in evaluation_data:
    for metric in evaluation_data[model]:
        if len(evaluation_data[model][metric]) >= 3:
            stats_result = calculate_stats(evaluation_data[model][metric])
            eval_results.append({
                'Model': model,
                'Metric': metric,
                'Mean': stats_result['mean'],
                'Std': stats_result['std'],
                'N': stats_result['n'],
                'Values': str(evaluation_data[model][metric])
            })

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv('eval/evaluation_statistics.csv', index=False)
print("✅ Evaluation statistics saved to: eval/evaluation_statistics.csv")

# 2. 학습 메트릭 통계 CSV
training_results = []
for model in training_metrics:
    for metric in training_metrics[model]:
        stats_result = calculate_stats(training_metrics[model][metric])
        training_results.append({
            'Model': model,
            'Metric': metric,
            'Mean': stats_result['mean'],
            'Std': stats_result['std'],
            'N': stats_result['n'],
            'Values': str(training_metrics[model][metric])
        })

training_df = pd.DataFrame(training_results)
training_df.to_csv('eval/training_statistics.csv', index=False)
print("✅ Training statistics saved to: eval/training_statistics.csv")

# 3. 모델 비교 결과 CSV
comparison_results = []

# HS 모델 비교 (Window 2 vs 5)
for metric in ['WordSim-353', 'SimLex-999', 'Google Analogy']:
    values1 = evaluation_data['hs_window-2_sub-True'][metric]
    values2 = evaluation_data['hs_window-5_sub-True'][metric]
    
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    # 효과 크기 (Cohen's d for paired samples)
    diff = np.array(values2) - np.array(values1)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    mean1, mean2 = np.mean(values1), np.mean(values2)
    
    # 통계적 유의성
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    comparison_results.append({
        'Comparison': 'HS Window-2 vs Window-5',
        'Metric': metric,
        'Model1_Mean': mean1,
        'Model2_Mean': mean2,
        'Difference': mean2 - mean1,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': significance,
        'better_model': 'Window-5' if mean2 > mean1 else 'Window-2'
    })

# 학습 메트릭 비교
for metric in ['loss', 'duration', 'gpu_memory']:
    if metric in training_metrics['hs_window-2_sub-True'] and metric in training_metrics['hs_window-5_sub-True']:
        values1 = training_metrics['hs_window-2_sub-True'][metric]
        values2 = training_metrics['hs_window-5_sub-True'][metric]
        
        t_stat, p_value = stats.ttest_rel(values1, values2)
        
        mean1, mean2 = np.mean(values1), np.mean(values2)
        
        if p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        comparison_results.append({
            'Comparison': 'HS Window-2 vs Window-5 (Training)',
            'Metric': metric,
            'Model1_Mean': mean1,
            'Model2_Mean': mean2,
            'Difference': mean2 - mean1,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': np.nan,  # 학습 메트릭은 효과 크기 계산 생략
            'significance': significance,
            'better_model': 'Window-2' if (metric == 'loss' and mean1 < mean2) or 
                           (metric == 'duration' and mean1 < mean2) else 'Window-5'
        })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('eval/model_comparisons.csv', index=False)
print("✅ Model comparisons saved to: eval/model_comparisons.csv")

# 4. 요약 테이블 CSV
summary_data = []
for metric in ['WordSim-353', 'SimLex-999', 'Google Analogy']:
    values1 = evaluation_data['hs_window-2_sub-True'][metric]
    values2 = evaluation_data['hs_window-5_sub-True'][metric]
    
    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    summary_data.append({
        'Metric': metric,
        'HS_Window2_Mean': mean1,
        'HS_Window2_Std': std1,
        'HS_Window5_Mean': mean2,
        'HS_Window5_Std': std2,
        'p_value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('eval/summary_table.csv', index=False)
print("✅ Summary table saved to: eval/summary_table.csv")

# 5. 효율성 분석 CSV
models = ['hs_window-2_sub-True', 'hs_window-5_sub-True', 'ns_window-2_sub-True', 'ns_window-5_sub-False']
performance_wordsim = [0.6726, 0.6990, 0.6428, 0.6392]  # WordSim-353 평균 성능
gpu_hours = [6.32, 12.72, 10.65, 10.35]  # duration 평균을 시간으로 변환

efficiency_data = []
for i, model in enumerate(models):
    if model in training_metrics:
        duration_mean = np.mean(training_metrics[model]['duration'])
        memory_mean = np.mean(training_metrics[model]['gpu_memory'])
        
        efficiency_data.append({
            'Model': model,
            'Performance_WordSim353': performance_wordsim[i],
            'GPU_Hours': duration_mean / 3600,
            'Memory_GB': memory_mean,
            'Performance_per_Hour': performance_wordsim[i] / (duration_mean / 3600),
            'Performance_per_GB': performance_wordsim[i] / memory_mean
        })

efficiency_df = pd.DataFrame(efficiency_data)
efficiency_df.to_csv('eval/efficiency_analysis.csv', index=False)
print("✅ Efficiency analysis saved to: eval/efficiency_analysis.csv")

print("\n📁 All CSV files saved in the 'eval/' directory:")
print("  - evaluation_statistics.csv")
print("  - training_statistics.csv") 
print("  - model_comparisons.csv")
print("  - summary_table.csv")
print("  - efficiency_analysis.csv")
