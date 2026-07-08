import re

fixes = {
    'tools.py': [
        ('description="要执行的统计动作。"', 'description="Statistical action to execute."'),
        ('description="描述统计或相关性分析所需列。"', 'description="Columns needed for descriptive statistics or correlation analysis."'),
        ('group_by: str | None = Field(default=None, description="分组汇总使用的Group column。")',
         'group_by: str | None = Field(default=None, description="Group column used for group summary.")'),
        ('description="分组汇总的指标定义。"', 'description="Metric definitions for group summary."'),
        ('description="分组汇总的排序列。"', 'description="Sort column for group summary."'),
        ('description="是否升序排序。"', 'description="Whether to sort in ascending order."'),
        ('description="分组汇总返回前几行。"', 'description="Number of top rows to return for group summary."'),
        ('group_col: str | None = Field(default=None, description="t 检验、ANOVA 或卡方检验的Group column。")',
         'group_col: str | None = Field(default=None, description="Group column for t-test, ANOVA, or chi-square test.")'),
        ('description="卡方检验的列 A。"', 'description="Column A for chi-square test."'),
        ('description="卡方检验的列 B。"', 'description="Column B for chi-square test."'),
        ('return result.startswith("代码执行失败:") or result.startswith("绘图执行失败:")',
         'return result.startswith("Code execution failed:") or result.startswith("Plot execution failed:")'),
        ('安全执行受限的数据分析代码，仅暴露白名单 helper API。',
         'Safely execute restricted data analysis code, exposing only whitelisted helper APIs.'),
        ('result = f"代码被安全策略拦截: {exc}"',
         'result = f"Code blocked by security policy: {exc}"'),
        ('安全执行绘图代码并保存生成的图片。',
         'Safely execute plotting code and save the generated image.'),
        ('result = f"绘图代码被安全策略拦截: {exc}"',
         'result = f"Plot code blocked by security policy: {exc}"'),
        ('直接训练 baseline 逻辑回归模型，并返回结构化 model_result。',
         'Train a baseline logistic regression model and return structured model_result.'),
        ('直接训练 baseline 线性回归模型，并返回结构化 model_result。',
         'Train a baseline linear regression model and return structured model_result.'),
        ('返回已有模型的 metrics_result。',
         'Return metrics_result for an existing model.'),
        ('返回已有模型的 feature_importance_result。',
         'Return feature_importance_result for an existing model.'),
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
