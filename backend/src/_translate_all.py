import glob, re

FILES = {}  # filename -> list of (old_chinese_text, new_english_text)

FILES['agent.py'] = [
    # Fix the remaining Chinese in the general_chat system prompt
    ('你是 Data Agent 的中文助手，目标是"准确、简洁、可执行"。请默认使用中文，语气专业友好，不要编造不存在的数据、接口或结论。如果用户问题属于普通聊天、概念解释、学习建议或通识问答，请直接回答。如果用户明确要求"基于已上传数据"的分析，但当前没有可用数据集，请明确提示先上传 CSV 文件。当信息不足以得出结论时，先说清缺失信息，再给最小可行下一步。', 
     'You are Data Agent, an AI data analysis assistant. Your goal: accurate, concise, actionable. Default to English. Be professional and friendly. Never fabricate data, APIs, or conclusions. If the user asks general chat, concept explanations, study advice, or common knowledge, answer directly. If the user explicitly requests analysis based on uploaded data but no dataset is available, clearly ask them to upload a CSV first. When information is insufficient to draw a conclusion, state what is missing and suggest the minimal next step.'),
]

FILES['chat_service.py'] = [
    ('最近一次结构化结果，供解释或跟进使用', 'Latest structured result for explanation or follow-up'),
    ('上游模型连接失败，请检查网络或模型服务配置。', 'Upstream model connection failed. Check network or model service configuration.'),
    ('上游模型服务返回异常状态。', 'Upstream model service returned an abnormal status.'),
    ('上游模型请求失败，请检查网络连接后重试。', 'Upstream model request failed. Check network connection and try again.'),
    ('服务器内部错误，请稍后重试。', 'Internal server error. Please try again later.'),
    ('模型调用失败：', 'Model call failed: '),
    ('请求参数不合法，请先输入问题。', 'Invalid request parameters. Please enter a question first.'),
    ('当前未选择数据集，请先上传 CSV 文件后再进行数据分析。', 'No dataset selected. Please upload a CSV file first.'),
]

FILES['conversation_context.py'] = [
    ('为什么', 'why'),
    ('逻辑回归', 'logistic regression'),
    ('线性回归', 'linear regression'),
]

FILES['data_manager.py'] = [
    ('分析预处理信息暂不可用', 'Analysis preprocessing info not available'),
    ('阶段未执行额外预处理', 'No additional preprocessing performed at this stage'),
    ('文件读取失败，请检查文件内容是否正确。', 'File read failed. Please check the file content.'),
    (' 请使用 ', ' Please use '),
]

FILES['dataset_recommendations.py'] = [
    ('的推荐问题生成器', 'recommended question generator'),
    ('请基于给定数据集摘要和', 'Based on the given dataset summary and '),
    ('个简洁', ' concise'),
    ('彼此不同的问题', 'distinct questions'),
    ('问题必须面向当前数据集', 'Questions must target the current dataset'),
    ('优先覆盖趋势', 'Prioritize covering trends'),
    ('建模中的高价值入口', 'High-value entry points for modeling'),
    ('若数据不适合建模', 'If the data is not suitable for modeling'),
    ('请不要强行给建模问题', 'Do not force modeling questions'),
    ('只输出', 'Output only '),
    ('格式为', 'Format: '),
    ('的趋势', ' trends'),
    ('并画一张折线图', 'and draw a line chart'),
]

FILES['ml_helpers.py'] = [
    ('目标列类别分布不满足当前测试集划分要求，请增加样本或调整', 'Target column class distribution does not meet current test split requirements. Increase samples or adjust '),
    ('训练集或测试集类别不足', 'Training or test set has insufficient classes'),
    ('无法稳定评估模型，', 'Cannot stably evaluate model, '),
    ('未能稳定提取模型系数', 'Failed to stably extract model coefficients'),
    ('有效样本不足，建议至少', 'Insufficient effective samples. Recommended at least '),
    ('行再训练线性回归模型', ' rows to retrain linear regression model'),
    ('目标列转换为数值后丢弃了', 'Target column after numeric conversion discarded '),
    ('未能稳定提取线性回归系数', 'Failed to stably extract linear regression coefficients'),
    ('当前模型未提供可解释的系数特征重要性', 'Current model does not provide interpretable coefficient-based feature importance'),
    ('当前还没有可复用的模型结果，请先训练', 'No reusable model result yet. Please train a model first'),
    ('必须为正整数', 'must be a positive integer'),
]

FILES['plan_executor.py'] = [
    ('当前结构化计划包含暂不支持的任务类型：', 'Current structured plan contains unsupported task type(s): '),
    ('已回退到通用分析执行', 'Fell back to general analysis execution'),
    ('暂不支持的任务类型', 'Unsupported task type(s)'),
    ('已按统一计划执行', 'Executed via unified plan'),
    ('当前先执行了', 'Currently executed '),
    ('个可直接落地的计划任务', ' directly executable plan tasks'),
    ('个更复杂任务暂未走结构化计划分支', ' more complex tasks not yet on structured plan path'),
    ('执行假设', 'Execution assumptions'),
    ('数据集概况', 'Dataset overview'),
    ('数据规模', 'Data size'),
    ('数值字段数', 'Numeric field count'),
    ('分类字段数', 'Categorical field count'),
    ('需要注意', 'Note'),
    ('转化率说明', 'Conversion rate note'),
    ('基于列', 'Based on column'),
]

FILES['plan_verifier.py'] = [
    ('未完成任务', 'Incomplete tasks'),
    ('缺少输出', 'Missing output'),
    ('计划未完整执行', 'Plan not fully executed'),
]

FILES['preprocessing.py'] = [
    ('没有可用于建模的特征列', 'No feature columns available for modeling'),
    ('阶段默认排除', 'Stage excluded by default: '),
    ('特征数超过上限，仅保留前', 'Feature count exceeds limit, keeping only top '),
    ('过滤后没有可用特征列', 'No usable feature columns after filtering'),
    # Note: target column exists but has missing values
    ('目标列存在缺失，已丢弃', 'Target column has missing values, dropped '),
    ('目标列有效样本为空', 'Target column effective samples are empty'),
    ('样本数超过上限，已截断为前', 'Sample count exceeds limit, truncated to top '),
    ('当前数据不包含可用于', 'Current data does not contain usable '),
]

FILES['routing_executor.py'] = [
    ('用中文写一段简洁、可信的数据集讲解', 'Write a concise, trustworthy dataset overview'),
    ('用中文明确指出缺失信息', 'Clearly state what information is missing'),
    ('你也可以直接点上方推荐问题，或者从这些方向开始：', 'You can also click the suggested questions above, or start from these directions:'),
    ('你先帮你快速讲解一下', 'Let me first give you a quick overview'),
    ('我也注意到几个数据质量注意事项', 'I have also noticed a few data quality notes'),
    ('的数据集概览助手', '\'s dataset overview assistant'),
    ('请基于给定的结构化上下文，用中文写一段简洁', 'Based on the given structured context, write a concise'),
]

FILES['routing_rules.py'] = [
    ('重要特征', 'important features'),
]

FILES['schema_profile.py'] = [
    ('可能可用作标签列', 'Potentially usable as label column'),
    ('二值列', 'Binary column'),
    ('二值分类列', 'Binary categorical column'),
]

FILES['self_correction.py'] = [
    ('请修复上一次执行失败的代码', 'Please fix the previously failed code'),
    ('尽量只做最小改动', 'Make minimal changes where possible'),
    ('不要使用', 'Do not use '),
    ('文件访问', 'File access'),
]

FILES['server.py'] = [
    ('请求参数不合法，请检查后重试。', 'Invalid request parameters. Please check and try again.'),
    ('服务器内部错误，请稍后重试。', 'Internal server error. Please try again later.'),
    ('已拒绝请求：', 'Request rejected: '),
    ('未安装或导入失败', 'Not installed or import failed'),
    ('开发环境将跳过', 'Development environment will skip'),
    ('路由注入', 'Route injection'),
    # Don't match "成功加载文件" everywhere, be specific
]

FILES['tools.py'] = [
    ('条形图', 'bar chart'),
    ('箱线图', 'box plot'),
    ('没有可用于相关性热力图的数值列', 'No numeric columns available for correlation heatmap'),
    ('相关性热力图', 'Correlation heatmap'),
    ('当前还没有可复用的数据理解结果', 'No reusable data understanding results yet'),
    ('当前没有可用数据集，请先上传数据', 'No active dataset. Please upload data first'),
    ('当前没有可用数据集，请先上传', 'No active dataset. Please upload'),
    ('后再训练模型', ' before retraining the model'),
    ('当前还没有可复用的统计结果，请先执行一次统计分析', 'No reusable statistical results yet. Run a statistical analysis first'),
    ('没有可用于数值描述统计的列', 'No columns available for numeric descriptive statistics'),
    ('没有可用于分类描述统计的列', 'No columns available for categorical descriptive statistics'),
    ('必须为', ' must be '),
]

count = 0
for filename, replacements in FILES.items():
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            count += 1
            print(f'  [{filename}] OK: {old[:40]}...')
        else:
            print(f'  [{filename}] MISS: {old[:40]}...')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print(f'\nTotal {count} replacements done')
