import re, glob

fixes = {
    'tools.py': [
        # Error messages
        ('return f"代码执行失败: {exc}"', 'return f"Code execution failed: {exc}"'),
        ('return "代码执行成功，但没有输出。请使用 print() 展示结果。"', 'return "Code executed successfully, but no output. Use print() to display results."'),
        ('return f"绘图执行失败: {exc}"', 'return f"Plot execution failed: {exc}"'),
        ('return "绘图代码执行完毕，但未生成图像对象。"', 'return "Plot code executed, but no image object was generated."'),
        ('raise SafeExecutionError("图片保存路径非法。")', 'raise SafeExecutionError("Image save path is invalid.")'),
        ('return "图表已生成"', 'return "Chart generated"'),
        ('raise SafeExecutionError(f"不允许访问敏感属性: {name}")', 'raise SafeExecutionError(f"Access to forbidden attribute: {name}")'),
        ('raise SafeExecutionError(f"不允许调用危险函数: {name}")', 'raise SafeExecutionError(f"Call to dangerous function: {name}")'),
        ('raise SafeExecutionError(f"df 当前仅支持只读访问，属性不可用: {name}")', 'raise SafeExecutionError(f"df is currently read-only, attribute not available: {name}")'),
        ('raise SafeExecutionError(f"不允许访问敏感属性: {name}")', 'raise SafeExecutionError(f"Access to forbidden attribute: {name}")'),
        ('raise SafeExecutionError(f"不允许调用危险函数: {name}")', 'raise SafeExecutionError(f"Call to dangerous function: {name}")'),
        ('raise SafeExecutionError(f"Series 当前仅支持只读访问，属性不可用: {name}")', 'raise SafeExecutionError(f"Series is currently read-only, attribute not available: {name}")'),
        # Pydantic Field descriptions
        ('py_code: str = Field(description="Python代码。可以使用变量 df、data、viz、stats、profile、ml。")',
         'py_code: str = Field(description="Python code. Available variables: df, data, viz, stats, profile, ml.")'),
        ('py_code: str = Field(description="绘图代码。需生成图像对象。")',
         'py_code: str = Field(description="Plotting code. Must generate an image object.")'),
        ('fname: str = Field(description="图像变量名，例如 \'fig\'。")',
         "fname: str = Field(description=\"Image variable name, e.g. 'fig'.\")"),
        ('target: str = Field(description="二分类目标列名称，例如 Churn。")',
         'target: str = Field(description="Binary classification target column name, e.g. Churn.")'),
        ('features: list[str] | None = Field(default=None, description="可选特征列名单。")',
         'features: list[str] | None = Field(default=None, description="Optional feature column list.")'),
        ('test_size: float | None = Field(default=None, description="可选测试集比例。")',
         'test_size: float | None = Field(default=None, description="Optional test set ratio.")'),
        ('positive_label: Any | None = Field(default=None, description="可选正类标签。")',
         'positive_label: Any | None = Field(default=None, description="Optional positive class label.")'),
        ('target: str = Field(description="数值回归目标列名称。")',
         'target: str = Field(description="Numeric regression target column name.")'),
        ('top_k: int = Field(default=10, description="返回前几个重要特征。")',
         'top_k: int = Field(default=10, description="Return top K important features.")'),
        ('description="训练动作时使用的模型类型。"',
         'description="Model type used for the training action."'),
        ('target: str | None = Field(default=None, description="训练动作的目标列。")',
         'target: str | None = Field(default=None, description="Target column for the training action.")'),
    ],
    'self_correction.py': [
        ('"没有可用于建模的Feature column"', '"No usable feature column for modeling"'),
        ('"安全策略拦截"', '"Blocked by security policy"'),
    ],
    'preprocessing.py': [
        ('return {"positive_label": None, "source": "ambiguous", "warning": "Could not reliably infer positive',
         'return {"positive_label": None, "source": "ambiguous", "warning": "Could not reliably infer positive class. Please provide positive_label explicitly."'),
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
            print(f'  [{filename}] OK: {old[:50]}...')
        else:
            print(f'  [{filename}] MISS: {old[:50]}...')
    if changed:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

print(f'\nTotal: {total} replacements')
