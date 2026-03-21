#!/usr/bin/env python3
"""runs.jsonl 콘솔 뷰어 - UTF-8 한글 지원"""
import json, sys, os, glob

def view_runs(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    records = [json.loads(line) for line in lines if line.strip()]
    if limit:
        records = records[:limit]
    
    for r in records:
        print(f"\n{'='*60}")
        print(f"  #{r['run_id']}  {r['prompt_label']}  {'OK' if r['success'] else 'FAIL'}")
        print(f"  {r['latency_ms']}ms | {r['tokens_generated']}tok | {r['tokens_per_sec']:.1f}tok/s")
        print(f"{'='*60}")
        print(f"\n[SYSTEM]\n{r.get('system_prompt','(없음)')}")
        print(f"\n[USER]\n{r.get('user_prompt','(없음)')}")
        print(f"\n[RESPONSE]\n{r.get('response','(없음)')}")
        if r.get('extra'):
            print(f"\n[EXTRA]\n{json.dumps(r['extra'], ensure_ascii=False, indent=2)}")
    
    print(f"\n--- {len(records)}건 표시 ---")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) < 2:
        # 인자 없으면 가장 최근 실험 찾기
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'experiments')
        dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        if not dirs:
            print("실험 데이터 없음"); sys.exit(1)
        path = os.path.join(dirs[-1], 'runs.jsonl')
        print(f"최근 실험: {os.path.basename(dirs[-1])}")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            path = os.path.join(path, 'runs.jsonl')
    
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    view_runs(path, limit)
