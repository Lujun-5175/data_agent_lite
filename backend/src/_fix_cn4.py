fixes = {
    'tools.py': [
        ('返回最近一次 ML 结构化结果。', 'Return the latest ML structured result.'),
        ('ML 请求请优先使用这个工具，不要先走 python_inter。', 'For ML requests, prefer this tool over python_inter.'),
        ('- action="train": 训练逻辑回归或线性回归\n- action="metrics": 返回模型指标\n- action="feature_importance": 返回特征重要性\n- action="latest": 返回最近一次 ML 结构化结果',
         '- action="train": Train logistic or linear regression\n- action="metrics": Return model metrics\n- action="feature_importance": Return feature importance\n- action="latest": Return the latest ML structured result'),
        ('result = "错误：训练动作必须提供 target。"', 'result = "Error: training action must provide target."'),
        ('统一的统计分析入口。统计类请求优先使用这个工具，而不是自己写 Python 代码。',
         'Unified statistical analysis entry point. Prefer this tool for statistics requests over writing custom Python code.'),
        ('result = "错误：correlation 需要至少提供两列。"', 'result = "Error: correlation requires at least two columns."'),
    ],
}

total = 0
for filename, replacements in fixes.items():
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
            total += 1
            print(f'  OK: {old[:50]}...')
        else:
            print(f'  MISS: {old[:50]}...')
    if changed:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

print(f'\nTotal: {total}')
