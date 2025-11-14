#!/usr/bin/env python3
"""
배치 평가 결과 분석 스크립트: 여러 모델의 평가 결과를 종합적으로 분석합니다.

사용법:
    python scripts/analyze_results.py --results_dir results/batch_eval
"""

import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def parse_model_info_from_filename(filename):
    """
    파일명에서 모델 정보를 추출합니다.
    
    예시: ns_window-5_sub-False__step-2000000_results.csv
    """
    stem = Path(filename).stem.replace('_results', '')
    
    info = {
        'mode': 'ns',
        'window': 5,
        'subsample': True,
        'seed': 42,
        'epoch': None,
        'step': None,
        'model_name': stem
    }
    
    # 모드 추출
    if stem.startswith('hs'):
        info['mode'] = 'hs'
    elif stem.startswith('ns'):
        info['mode'] = 'ns'
    
    # window 크기 추출
    window_match = re.search(r'window-(\d+)', stem)
    if window_match:
        info['window'] = int(window_match.group(1))
    
    # subsample 설정 추출
    if 'sub-False' in stem:
        info['subsample'] = False
    elif 'sub-True' in stem:
        info['subsample'] = True
    
    # seed 추출
    seed_match = re.search(r'seed-(\d+)', stem)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
    
    # epoch 또는 step 추출
    epoch_match = re.search(r'epoch-(\d+)', stem)
    if epoch_match:
        info['epoch'] = int(epoch_match.group(1))
    
    step_match = re.search(r'step-(\d+)', stem)
    if step_match:
        info['step'] = int(step_match.group(1))
    
    return info

def load_all_results(results_dir):
    """
    결과 디렉토리에서 모든 CSV 파일을 로드하고 통합합니다.
    """
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    all_results = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            model_info = parse_model_info_from_filename(csv_file)
            
            # 각 행에 모델 정보 추가
            for _, row in df.iterrows():
                result_row = {
                    'model_name': model_info['model_name'],
                    'mode': model_info['mode'],
                    'window': model_info['window'],
                    'subsample': model_info['subsample'],
                    'seed': model_info['seed'],
                    'epoch': model_info['epoch'],
                    'step': model_info['step'],
                    'dataset': row['Dataset'],
                    'metric': row['Metric'],
                    'score': float(row['Score'])
                }
                all_results.append(result_row)
                
        except Exception as e:
            print(f"❌ {csv_file} 로드 실패: {e}")
    
    return pd.DataFrame(all_results)

def create_comparison_plots(df, output_dir):
    """
    비교 분석을 위한 시각화를 생성합니다.
    """
    plt.style.use('default')
    fig_size = (15, 10)
    
    # 1. 데이터셋별 성능 비교 (모드별)
    plt.figure(figsize=fig_size)
    
    datasets = df['dataset'].unique()
    modes = df['mode'].unique()
    
    for i, dataset in enumerate(datasets, 1):
        plt.subplot(2, 2, i)
        
        dataset_df = df[df['dataset'] == dataset]
        
        # 모드별 박스플롯
        sns.boxplot(data=dataset_df, x='mode', y='score', hue='window')
        plt.title(f'{dataset} Performance by Mode and Window Size')
        plt.ylabel('Score')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_mode_window.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Subsample 효과 분석
    plt.figure(figsize=fig_size)
    
    for i, dataset in enumerate(datasets, 1):
        plt.subplot(2, 2, i)
        
        dataset_df = df[df['dataset'] == dataset]
        
        sns.boxplot(data=dataset_df, x='subsample', y='score', hue='mode')
        plt.title(f'{dataset}: Subsampling Effect')
        plt.ylabel('Score')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subsampling_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 전체 성능 히트맵
    plt.figure(figsize=(12, 8))
    
    # 피벗 테이블 생성 (평균 점수)
    pivot_df = df.groupby(['mode', 'window', 'subsample', 'dataset'])['score'].mean().reset_index()
    
    # 각 데이터셋별로 히트맵 생성
    for i, dataset in enumerate(datasets, 1):
        plt.subplot(2, 2, i)
        
        dataset_pivot = pivot_df[pivot_df['dataset'] == dataset]
        heatmap_data = dataset_pivot.pivot_table(
            values='score', 
            index=['mode', 'subsample'], 
            columns='window'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'{dataset} Average Scores')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_dir):
    """
    종합 분석 리포트를 생성합니다.
    """
    report_path = os.path.join(output_dir, 'analysis_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Word2Vec 모델 평가 결과 분석 리포트\n\n")
        
        # 기본 통계
        f.write("## 📊 기본 통계\n\n")
        f.write(f"- 총 평가된 모델 수: {df['model_name'].nunique()}\n")
        f.write(f"- 평가 데이터셋: {', '.join(df['dataset'].unique())}\n")
        f.write(f"- 훈련 모드: {', '.join(df['mode'].unique())}\n")
        f.write(f"- 윈도우 크기: {', '.join(map(str, sorted(df['window'].unique())))}\n\n")
        
        # 최고 성능 모델들
        f.write("## 🏆 최고 성능 모델들\n\n")
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best_model = dataset_df.loc[dataset_df['score'].idxmax()]
            
            f.write(f"### {dataset}\n")
            f.write(f"- **모델**: {best_model['model_name']}\n")
            f.write(f"- **점수**: {best_model['score']:.4f}\n")
            f.write(f"- **설정**: {best_model['mode'].upper()}, window={best_model['window']}, subsample={best_model['subsample']}\n\n")
        
        # 모드별 평균 성능
        f.write("## 📈 모드별 평균 성능\n\n")
        
        mode_performance = df.groupby(['mode', 'dataset'])['score'].mean().reset_index()
        
        for dataset in df['dataset'].unique():
            f.write(f"### {dataset}\n")
            dataset_perf = mode_performance[mode_performance['dataset'] == dataset]
            
            for _, row in dataset_perf.iterrows():
                f.write(f"- **{row['mode'].upper()}**: {row['score']:.4f}\n")
            f.write("\n")
        
        # 윈도우 크기별 성능
        f.write("## 🔍 윈도우 크기별 성능\n\n")
        
        window_performance = df.groupby(['window', 'dataset'])['score'].mean().reset_index()
        
        for dataset in df['dataset'].unique():
            f.write(f"### {dataset}\n")
            dataset_perf = window_performance[window_performance['dataset'] == dataset]
            
            for _, row in dataset_perf.iterrows():
                f.write(f"- **Window {row['window']}**: {row['score']:.4f}\n")
            f.write("\n")
        
        # Subsampling 효과
        f.write("## ⚡ Subsampling 효과\n\n")
        
        subsample_performance = df.groupby(['subsample', 'dataset'])['score'].mean().reset_index()
        
        for dataset in df['dataset'].unique():
            f.write(f"### {dataset}\n")
            dataset_perf = subsample_performance[subsample_performance['dataset'] == dataset]
            
            for _, row in dataset_perf.iterrows():
                subsample_str = "ON" if row['subsample'] else "OFF"
                f.write(f"- **Subsample {subsample_str}**: {row['score']:.4f}\n")
            f.write("\n")
    
    print(f"📋 분석 리포트 저장됨: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="배치 평가 결과 분석")
    parser.add_argument("--results_dir", default="results/batch_eval", help="결과 CSV 파일들이 있는 디렉토리")
    parser.add_argument("--output_dir", default="results/analysis", help="분석 결과를 저장할 디렉토리")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📁 결과 로딩 중: {args.results_dir}")
    
    # 모든 결과 로드
    df = load_all_results(args.results_dir)
    
    if df.empty:
        print("❌ 로드된 결과가 없습니다. 결과 디렉토리를 확인해주세요.")
        return
    
    print(f"✅ {len(df)}개의 결과를 로드했습니다.")
    print(f"📊 {df['model_name'].nunique()}개의 고유 모델")
    
    # 통합 결과 저장
    combined_path = os.path.join(args.output_dir, 'combined_results.csv')
    df.to_csv(combined_path, index=False)
    print(f"💾 통합 결과 저장됨: {combined_path}")
    
    # 시각화 생성
    print("📈 시각화 생성 중...")
    create_comparison_plots(df, args.output_dir)
    
    # 분석 리포트 생성
    print("📋 분석 리포트 생성 중...")
    generate_summary_report(df, args.output_dir)
    
    print(f"\n✅ 분석 완료! 결과는 {args.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()

