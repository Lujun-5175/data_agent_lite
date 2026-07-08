import re, glob

fixes = {
    'agent.py': [
        ('若用户显式要求模型指标或特征重要性，应按需继续调用 ', 'If the user explicitly requests model metrics or feature importance, continue calling '),
        ('并分别使用 ', 'with '),
        ('，复用最近的模型 artifact。', ', reusing the latest model artifact.'),
    ],
    'data_manager.py': [
        ('数据集不存在或已被删除。', 'Dataset not found or has been deleted.'),
        ('CSV 文件格式不正确，无法解析。', 'Invalid CSV format, cannot parse.'),
        ('指定的列不存在。', 'Specified column does not exist.'),
    ],
    'api_models.py': [
        ('第一列名称', 'First column name'),
        ('第二列名称', 'Second column name'),
    ],
    'chat_service.py': [
        ('会话历史压缩失败，请重新发起请求。', 'Conversation history compression failed. Please send a new request.'),
    ],
    'preprocessing.py': [
        ('，已截断为前 ', ', truncated to top '),
        ('目标列为空，Cannot infer positive class label。', 'Target column is empty. Cannot infer positive class label.'),
        ('按负类标签 ', 'Inferred from negative label '),
        (' 反推正类 ', ', positive class = '),
    ],
    'plan_executor.py': [
        ('计算，正类取值为', 'computed, positive class ='),
    ],
    'schema_profile.py': [
        ('（可解析比例 ', ' (parsable ratio: '),
        ('). ', ').'),
    ],
    'self_correction.py': [
        ('import、File access、eval、exec 或任何被禁止的 API。', 'import, File access, eval, exec, or any other forbidden APIs.'),
        ('代码执行失败', 'Code execution failed'),
        ('绘图执行失败', 'Plot execution failed'),
        ('列不存在', 'Column does not exist'),
        ('排序列不存在', 'Sort column does not exist'),
        ('目标列', 'Target column'),
        ('不适合建模', 'Not suitable for modeling'),
        ('特征列', 'Feature column'),
        ('分组列', 'Group column'),
        ('列名字符串相似', 'Column name string similarity'),
        ('请修复上一次执行失败的代码', 'Please fix the previously failed code'),
        ('尽量只做最小改动', 'Make minimal changes'),
    ],
    'server.py': [
        ('Upload path is not safe，已拒绝请求。', 'Upload path is not safe. Request rejected.'),
        ('File upload failed，请稍后重试。', 'File upload failed. Please try again later.'),
    ],
    'ml_helpers.py': [
        ('，无法稳定评估模型.', '. Cannot stably evaluate model.'),
        ('指定的模型 artifact 不存在.', 'Specified model artifact does not exist.'),
        ('指定的模型 artifact 不属于当前数据集.', 'Specified model artifact does not belong to current dataset.'),
        ('当前没有可用模型结果，请先训练 baseline 模型.', 'No model results available. Please train a baseline model first.'),
        ('样本量过小，无法进行分层训练/测试划分.', 'Sample size too small for stratified train/test split.'),
        ('目标列需要恰好两类才能进行分层二分类训练.', 'Target column needs exactly 2 classes for stratified binary classification.'),
        ('当前 test_size 导致训练集或测试集过小，无法稳定训练.', 'Current test_size makes train/test sets too small for stable training.'),
        ('目标列类别分布不平衡且少数类样本不足，无法进行分层划分.', 'Target class distribution imbalanced with insufficient minority samples for stratified split.'),
    ],
    'dataset_recommendations.py': [
        ('按 ', 'Group by '),
        (' 分组比较 ', ', compare '),
        (' 的差异，并给出结论。', ' differences and give conclusions.'),
        (' 和 ', ' and '),
        (' 的相关性是多少？', ' correlation?'),
        (' 预测 ', ' to predict '),
        ('，跑一个线性回归并汇报指标。', ', run linear regression and report metrics.'),
        ('，尝试一个 baseline 分类模型并汇报指标。', ', try a baseline classification model and report metrics.'),
        ('请先做一份描述性统计，并指出最值得关注的字段。', 'Run descriptive statistics and highlight notable fields.'),
        ('按一个关键分类字段分组，比较主要指标差异。', 'Group by a key categorical field and compare major metric differences.'),
        ('哪些字段最适合继续做相关性、检验或建模分析？', 'Which fields are best for correlation, testing, or modeling?'),
    ],
    'tools.py': [
        ('当前没有可用数据集。请先上传数据。', 'No active dataset. Please upload data first.'),
        ('当前没有可用数据集。请先上传 CSV  before retraining the model。', 'No active dataset. Please upload CSV before retraining the model.'),
        ('当前还没有可复用的统计结果。请先执行一次统计分析。', 'No reusable statistical results yet. Please run a statistical analysis first.'),
        ('样本量过小，t test results may be unstable（Recommended at least per group 2 条）。', 'Sample size too small, t test results may be unstable (recommended at least 2 per group).'),
        ('Some expected frequencies are less than 5，Chi-square test assumptions are weak，请谨慎解释。', 'Some expected frequencies are less than 5. Chi-square test assumptions are weak. Interpret with caution.'),
        (' 样本量小于 2，results may be unstable。', ' sample size < 2, results may be unstable.'),
        ('ANOVA 至少需要 3 个有效分组。', 'ANOVA requires at least 3 valid groups.'),
        ('以下列为数值列，已忽略：', 'The following columns are numeric and were ignored: '),
        ('，仅保留前 ', ', keeping only top '),
        ('列为空，无法推断正类标签；rate 结果将为空。', 'Column is empty. Cannot infer positive class label; rate result will be empty.'),
        ('列包含二值数值 ', 'Column has binary numeric values '),
        ('，按较大值作为正类。', ', using larger value as positive class.'),
        ('rate 仅建议用于二值标签列，当前列无法可靠推断正类。', 'Rate is only recommended for binary label columns. Current column cannot reliably infer positive class.'),
        ('字符串标签为 ', 'String labels: '),
        (' 类，无法可靠推断正类；rate 结果将为空。', ' classes. Cannot reliably infer positive class; rate result will be empty.'),
        ('按负类标签 ', 'Inferred from negative label '),
        (' 反推正类 ', ', positive class = '),
        ('未能在标签 ', 'Could not identify a stable positive class among labels '),
        (' 中识别稳定正类，请显式提供 positive_label。', '. Please provide positive_label explicitly.'),
        ('样本不足，无法稳定判断统计显著性。', 'Insufficient samples to stably determine statistical significance.'),
        ('差异/关联较显著（p < 0.05）。', 'Difference/association is statistically significant (p < 0.05).'),
        ('未观察到显著差异/关联（p >= 0.05）。', 'No significant difference/association observed (p >= 0.05).'),
        ('自动选择分类列超过 ', 'Auto-selected categorical columns exceeded '),
    ],
    'routing_executor.py': [
        ('必须覆盖：数据规模、字段大类、值得注意的 warning、Suggestions on where to start。', 'Must cover: data size, field categories, notable warnings, suggestions on where to start.'),
        ('输出纯文本，不要输出 JSON。', 'Output plain text, not JSON.'),
        ('数据集摘要：', 'Dataset Summary:'),
        ('Tool execution failed，请稍后重试。', 'Tool execution failed. Please try again later.'),
        ('The same analysis tool has been called too many times，System has stopped this loop。请缩小范围后重试。', 'The same analysis tool has been called too many times. System has stopped this loop. Please narrow scope and retry.'),
        ('任务连续多步没有产生有效进展，系统已主动停止。请改成更具体的问题后重试。', 'Multiple consecutive steps without effective progress. System has stopped. Please rephrase with a more specific question.'),
        ('本次建模请求没有调用直接的 ml 工具，请先通过 ml_execute 完成建模。', 'This modeling request did not call a direct ml tool. Please use ml_execute to complete modeling first.'),
        ('f"本次建模请求缺少结构化结果：', 'f"This modeling request is missing structured results: '),
        ('结构化计划未完整执行。', 'Structured plan was not fully executed.'),
        ('本次图表请求没有成功生成可展示的图片结果，请检查字段名或图表描述后重试。', 'This chart request did not generate a displayable image. Please check field names or chart description and retry.'),
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
            print(f'  [{filename}] OK: {old[:40]}')
        else:
            print(f'  [{filename}] MISS: {old[:40]}')
    if changed:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

print(f'\nTotal: {total} replacements')
