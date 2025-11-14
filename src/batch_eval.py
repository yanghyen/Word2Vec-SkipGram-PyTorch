#!/usr/bin/env python3
"""
배치 평가 스크립트: runs/eval 디렉토리의 모든 .pth 파일들을 자동으로 평가합니다.

사용법:
    python scripts/batch_eval.py [--output_dir results] [--pattern "*epoch-1.pth"]
"""

import os
import glob
import subprocess
import argparse
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

def parse_checkpoint_name(checkpoint_path):
    """
    체크포인트 파일명에서 설정 정보를 추출합니다.
    
    예시:
    - ns_window-5_sub-False__step-2000000.pth -> ns, window=5, subsample=False
    - hs_epoch-1.pth -> hs, epoch=1
    """
    filename = Path(checkpoint_path).stem
    
    # 기본값
    info = {
        'mode': 'ns',  # ns 또는 hs
        'window': 5,
        'subsample': True,
        'seed': 42,
        'epoch': None,
        'step': None
    }
    
    # 모드 추출 (ns 또는 hs)
    if filename.startswith('hs'):
        info['mode'] = 'hs'
    elif filename.startswith('ns'):
        info['mode'] = 'ns'
    
    # window 크기 추출
    window_match = re.search(r'window-(\d+)', filename)
    if window_match:
        info['window'] = int(window_match.group(1))
    
    # subsample 설정 추출
    if 'sub-False' in filename or 'subsample-off' in filename:
        info['subsample'] = False
    elif 'sub-True' in filename or 'subsample-on' in filename:
        info['subsample'] = True
    
    # seed 추출
    seed_match = re.search(r'seed-(\d+)', filename)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
    
    # epoch 또는 step 추출
    epoch_match = re.search(r'epoch-(\d+)', filename)
    if epoch_match:
        info['epoch'] = int(epoch_match.group(1))
    
    step_match = re.search(r'step-(\d+)', filename)
    if step_match:
        info['step'] = int(step_match.group(1))
    
    return info

def find_matching_config(checkpoint_info, configs_dir="configs"):
    """
    체크포인트 정보에 맞는 config 파일을 찾습니다.
    """
    mode = checkpoint_info['mode']
    window = checkpoint_info['window']
    subsample = 'on' if checkpoint_info['subsample'] else 'off'
    seed = checkpoint_info['seed']
    
    # 가능한 config 파일명 패턴들
    patterns = [
        f"{mode}_window-{window}_subsample-{subsample}_seed-{seed}.yaml",
        f"{mode}_window-{window}_subsample-{subsample}_seed-42.yaml",  # fallback to seed 42
        f"{mode}_window-{window}_subsample-{subsample}.yaml",  # seed 없는 버전
    ]
    
    for pattern in patterns:
        config_path = os.path.join(configs_dir, pattern)
        if os.path.exists(config_path):
            return config_path
    
    # 찾지 못한 경우 가장 유사한 것을 찾기
    config_files = glob.glob(f"{configs_dir}/{mode}_window-{window}_*.yaml")
    if config_files:
        return config_files[0]  # 첫 번째 매칭되는 파일 반환
    
    return None

def run_evaluation(config_path, checkpoint_path, output_csv=None):
    """
    단일 체크포인트에 대해 평가를 실행합니다.
    """
    # 고정된 데이터셋 경로들
    wordsim_csv = "data/word_similarity/combined.csv"
    simlex_txt = "data/word_similarity/SimLex-999/SimLex-999.txt"
    analogy_txt = "data/word_similarity/word2vec/trunk/questions-words.txt"
    
    # eval.py 실행 명령 구성
    cmd = [
        "python", "src/eval.py",
        config_path,
        checkpoint_path,
        wordsim_csv,
        simlex_txt,
        analogy_txt
    ]
    
    if output_csv:
        cmd.extend(["--save_csv", output_csv])
    
    print(f"🚀 실행 중: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ 성공: {checkpoint_path}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ 실패: {checkpoint_path}")
        print(f"에러: {e.stderr}")
        return False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="배치 평가 스크립트")
    parser.add_argument("--eval_dir", default="runs/eval", help="평가할 .pth 파일들이 있는 디렉토리")
    parser.add_argument("--configs_dir", default="configs", help="config 파일들이 있는 디렉토리")
    parser.add_argument("--output_dir", default="results/batch_eval", help="결과를 저장할 디렉토리")
    parser.add_argument("--pattern", default="*epoch-1.pth", help="평가할 파일 패턴 (예: *epoch-1.pth)")
    parser.add_argument("--dry_run", action="store_true", help="실제 실행하지 않고 계획만 출력")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # .pth 파일들 찾기
    pattern_path = os.path.join(args.eval_dir, "**", args.pattern)
    checkpoint_files = glob.glob(pattern_path, recursive=True)
    
    print(f"📁 {len(checkpoint_files)}개의 체크포인트 파일을 찾았습니다:")
    
    evaluation_plan = []
    
    for checkpoint_path in sorted(checkpoint_files):
        print(f"\n📄 분석 중: {checkpoint_path}")
        
        # 체크포인트 정보 추출
        checkpoint_info = parse_checkpoint_name(checkpoint_path)
        print(f"   정보: {checkpoint_info}")
        
        # 매칭되는 config 파일 찾기
        config_path = find_matching_config(checkpoint_info, args.configs_dir)
        
        if config_path:
            print(f"   ✅ Config 찾음: {config_path}")
            
            # 출력 CSV 파일명 생성
            checkpoint_name = Path(checkpoint_path).stem
            output_csv = os.path.join(args.output_dir, f"{checkpoint_name}_results.csv")
            
            evaluation_plan.append({
                'checkpoint': checkpoint_path,
                'config': config_path,
                'output_csv': output_csv,
                'info': checkpoint_info
            })
        else:
            print(f"   ❌ 매칭되는 config 파일을 찾을 수 없습니다")
    
    print(f"\n📊 총 {len(evaluation_plan)}개의 평가를 실행할 예정입니다.")
    
    if args.dry_run:
        print("\n🔍 DRY RUN 모드 - 실제 실행하지 않습니다:")
        for i, plan in enumerate(evaluation_plan, 1):
            print(f"\n{i}. {plan['checkpoint']}")
            print(f"   Config: {plan['config']}")
            print(f"   Output: {plan['output_csv']}")
        return
    
    # 실제 평가 실행
    results_summary = []
    successful = 0
    failed = 0
    
    for i, plan in enumerate(evaluation_plan, 1):
        print(f"\n{'='*60}")
        print(f"평가 {i}/{len(evaluation_plan)}: {Path(plan['checkpoint']).name}")
        print(f"{'='*60}")
        
        success, output = run_evaluation(
            plan['config'], 
            plan['checkpoint'], 
            plan['output_csv']
        )
        
        if success:
            successful += 1
            results_summary.append({
                'checkpoint': plan['checkpoint'],
                'config': plan['config'],
                'status': 'SUCCESS',
                'output_csv': plan['output_csv']
            })
        else:
            failed += 1
            results_summary.append({
                'checkpoint': plan['checkpoint'],
                'config': plan['config'],
                'status': 'FAILED',
                'error': output
            })
    
    # 최종 요약
    print(f"\n{'='*60}")
    print(f"배치 평가 완료!")
    print(f"{'='*60}")
    print(f"✅ 성공: {successful}개")
    print(f"❌ 실패: {failed}개")
    print(f"📁 결과 저장 위치: {args.output_dir}")
    
    # 요약 파일 저장
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.output_dir, f"batch_eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"📋 요약 파일: {summary_path}")

if __name__ == "__main__":
    main()

