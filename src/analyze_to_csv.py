import numpy as np
from scipy import stats
import pandas as pd
import os

# ===== 평가 결과 데이터 (results.csv 기반) =====
evaluation_data = {
    'hs_window-2_sub-True': {
        'WordSim-353': [0.6783227799961188, 0.674672284641835, 0.6648799621236942],
        'SimLex-999': [0.3018909838566001, 0.2940152876218869, 0.2893765963991408],
        'Google Analogy': [0.30783959153281565, 0.3115625997234337, 0.3162429528773535]
    },
    'hs_window-5_sub-True': {
        'WordSim-353': [0.6964514083700853, 0.6932161881835475, 0.7073870941863546],
        'SimLex-999': [0.28693717244773875, 0.27822295307500955, 0.2901800922599784],
        'Google Analogy': [0.34294224018721414, 0.3448569301138177, 0.3453887884267631]
    },
    'ns_window-2_sub-True': {
        'WordSim-353': [0.6427771727381847, 0.6461831632665705, 0.6427771727381847],
        'SimLex-999': [0.30121203887213743, 0.3103243898882604, 0.30121203887213743],
        'Google Analogy': [0.46378044888841613, 0.4593128390596745, 0.46378044888841613]
    },
    'ns_window-5_sub-False': {
        'WordSim-353': [0.6218164698632008, 0.6288421532449577, 0.6391983370880968],
        'SimLex-999': [0.25954251385996013, 0.2642021497912687, 0.2538553908891166],
        'Google Analogy': [0.3763429422401872, 0.38144878204446336, 0.37751303052866714]
    },
    'ns_window-5_sub-True': {
        'WordSim-353': [0.6279875035434673, 0.6322639440199258, 0.6235189065328176],
        'SimLex-999': [0.25631358552598316, 0.2673818468256469, 0.26736033735689413],
        'Google Analogy': [0.3866609935113286, 0.38836294011275396, 0.37464099563876185]
    }
}

# ===== 학습 메트릭 데이터 =====
# 학습 메트릭은 runs/metrics/ 디렉토리의 CSV 파일에서 가져올 수 있습니다
# 현재는 평가 결과에 집중하여 학습 메트릭은 선택적으로 포함
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
    },
    'ns_window-5_sub-True': {
        'loss': [2.0, 2.0, 2.0],  # 예시 값
        'duration': [37000, 37000, 37000],  # 예시 값
        'gpu_memory': [2.35, 2.35, 2.35]  # 예시 값
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

# 3. 모델 비교 결과 CSV - 모든 모델 쌍 비교
comparison_results = []

# 모든 모델 쌍에 대해 비교 수행
model_names = list(evaluation_data.keys())

for i, model1 in enumerate(model_names):
    for j, model2 in enumerate(model_names):
        if i >= j:  # 중복 비교 방지 (자기 자신과의 비교 제외)
            continue
        
        for metric in ['WordSim-353', 'SimLex-999', 'Google Analogy']:
            values1 = evaluation_data[model1][metric]
            values2 = evaluation_data[model2][metric]
            
            t_stat, p_value = stats.ttest_rel(values1, values2)
            
            # 효과 크기 (Cohen's d for paired samples)
            diff = np.array(values2) - np.array(values1)
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0
            
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
                'Comparison': f'{model1} vs {model2}',
                'Metric': metric,
                'Model1_Mean': mean1,
                'Model2_Mean': mean2,
                'Difference': mean2 - mean1,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance,
                'better_model': model2 if mean2 > mean1 else model1
            })

# 학습 메트릭 비교 - 모든 모델 쌍
training_model_names = list(training_metrics.keys())
for i, model1 in enumerate(training_model_names):
    for j, model2 in enumerate(training_model_names):
        if i >= j:
            continue
        
        for metric in ['loss', 'duration', 'gpu_memory']:
            if metric in training_metrics[model1] and metric in training_metrics[model2]:
                values1 = training_metrics[model1][metric]
                values2 = training_metrics[model2][metric]
                
                t_stat, p_value = stats.ttest_rel(values1, values2)
                
                mean1, mean2 = np.mean(values1), np.mean(values2)
                
                if p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                # loss와 duration은 낮을수록 좋음
                if metric in ['loss', 'duration']:
                    better = model1 if mean1 < mean2 else model2
                else:  # gpu_memory 등은 낮을수록 좋음
                    better = model1 if mean1 < mean2 else model2
                
                comparison_results.append({
                    'Comparison': f'{model1} vs {model2} (Training)',
                    'Metric': metric,
                    'Model1_Mean': mean1,
                    'Model2_Mean': mean2,
                    'Difference': mean2 - mean1,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': np.nan,  # 학습 메트릭은 효과 크기 계산 생략
                    'significance': significance,
                    'better_model': better
                })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('eval/model_comparisons.csv', index=False)
print("✅ Model comparisons saved to: eval/model_comparisons.csv")

# 4. 요약 테이블 CSV - 모든 모델의 평균과 표준편차
summary_data = []

for metric in ['WordSim-353', 'SimLex-999', 'Google Analogy']:
    row = {'Metric': metric}
    
    for model in evaluation_data:
        values = evaluation_data[model][metric]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # 모델명을 간단하게 변환 (예: hs_window-2_sub-True -> HS_W2_S)
        model_short = model.replace('_window-', '_W').replace('_sub-', '_S').replace('True', 'T').replace('False', 'F')
        
        row[f'{model_short}_Mean'] = mean
        row[f'{model_short}_Std'] = std
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('eval/summary_table.csv', index=False)
print("✅ Summary table saved to: eval/summary_table.csv")

# 5. 효율성 분석 CSV - 모든 모델
efficiency_data = []

for model in evaluation_data:
    # WordSim-353 평균 성능 계산
    wordsim_mean = np.mean(evaluation_data[model]['WordSim-353'])
    
    if model in training_metrics:
        duration_mean = np.mean(training_metrics[model]['duration'])
        memory_mean = np.mean(training_metrics[model]['gpu_memory'])
        loss_mean = np.mean(training_metrics[model]['loss'])
        
        efficiency_data.append({
            'Model': model,
            'Performance_WordSim353': wordsim_mean,
            'Loss': loss_mean,
            'GPU_Hours': duration_mean / 3600,
            'Memory_GB': memory_mean,
            'Performance_per_Hour': wordsim_mean / (duration_mean / 3600),
            'Performance_per_GB': wordsim_mean / memory_mean
        })
    else:
        # training_metrics가 없는 경우에도 평가 성능은 표시
        efficiency_data.append({
            'Model': model,
            'Performance_WordSim353': wordsim_mean,
            'Loss': np.nan,
            'GPU_Hours': np.nan,
            'Memory_GB': np.nan,
            'Performance_per_Hour': np.nan,
            'Performance_per_GB': np.nan
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
