# Data Agent - AI 数据分析助手

一个基于 LangChain 1.1 和 React 的智能数据分析应用，支持 CSV 数据上传、AI 对话分析和数据可视化。

## 主要功能

### 数据管理

- **CSV 文件上传**：支持在聊天输入区上传 CSV 文件
- **数据预览**：显示数据预览与列类型识别
- **多层数据语义**：保留 `raw_df`，按需生成 `analysis_df`（对外兼容为 `working_df`）
- **Schema Profiling**：上传后生成结构化 `schema_profile`，包含列语义、可用性标记与风险提示
- **预处理透明化**：分析阶段生成 `preprocess_result`，记录步骤、保留/排除列与 warning
- **建模准备规划（仅规划）**：按需生成 `model_prep_plan`，用于目标候选与特征可用性说明

### AI 数据分析

- **流式对话**：基于 Server-Sent Events (SSE) 的实时流式响应
- **受限代码执行**：Agent 通过白名单 helper API 执行受限分析/绘图代码
- **安全保护**：静态 AST 校验 + 执行超时 + 禁止文件/网络写出
- **统计分析 helper**：支持描述统计、分组汇总、相关性、t 检验、卡方检验、ANOVA
- **Baseline ML helper**：支持受限逻辑回归/线性回归训练、指标查询、系数型特征重要性
- **图表生成**：图像统一写入后端 `static/images` 目录
- **动态上下文**：根据当前数据集自动更新 Agent 的 System Prompt

### 数据可视化

- **智能图表**：根据变量类型自动生成合适的图表
  - 离散 + 离散 → 堆叠柱状图
  - 连续 + 连续 → 散点图
  - 混合类型 → 分组柱状图
- **相关性分析**：点击变量快速计算相关系数
- **图表展示**：Agent 生成的图表会插入聊天消息流中


## 项目结构

```
Data Agent/
├── backend/                 # 后端服务
│   ├── src/
│   │   ├── agent.py        # LangGraph Agent 定义
│   │   ├── server.py       # FastAPI 服务器
│   │   ├── data_manager.py # 数据管理模块
│   │   ├── tools.py        # Agent 工具（Python执行、绘图）
│   │   └── state.py        # Agent 状态定义
│   ├── static/             # 静态文件（生成的图片）
│   ├── temp_data/          # 临时上传的数据文件
│   └── pyproject.toml      # Python 依赖配置
│
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/     # React 组件
│   │   │   ├── Header.tsx           # 顶部导航栏
│   │   │   ├── ChatInterface.tsx    # AI 对话界面
│   │   │   ├── DataUpload.tsx       # 数据上传组件
│   │   │   ├── VisualizationPanel.tsx # 可视化面板
│   │   │   └── ui/                  # UI 组件库
│   │   ├── config/
│   │   │   └── api.ts      # API 配置
│   │   └── App.tsx         # 主应用组件
│   ├── public/             # 静态资源
│   └── package.json        # Node.js 依赖配置
│
└── README.md              # 项目说明文档
```

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- DeepSeek API Key（在 [DeepSeek 官网](https://www.deepseek.com/) 获取）

### 安装步骤

#### 1. 配置后端

```bash
cd backend

# 安装依赖（需要先安装 uv，参考 https://uv.fan/w/installation）
uv sync

# 配置环境变量
# 在 backend 目录下创建 .env 文件
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
```

#### 2. 配置前端

```bash
cd frontend

# 安装依赖
npm install
```

### 启动服务

#### 启动后端

```bash
cd backend
uv run python -m src.server
```

后端服务将在 `http://localhost:8002` 启动。

#### 启动前端

```bash
cd frontend
npm run dev
```

前端应用将在 `http://localhost:5173` 启动（Vite 默认端口）。

### 访问应用

打开浏览器访问 `http://localhost:5173`，即可开始使用。

## 配置说明
> 安全说明：当前版本采用“受限执行 + AST 校验 + 超时保护”的实用防护模型，
> 可显著降低风险，但它不是操作系统级强隔离沙箱。
> 生产环境仍建议在容器、最小权限和网络隔离策略下部署。

### 后端配置

**环境变量**（`.env` 文件）：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
# 可选：生产环境显式 CORS 白名单（逗号分隔）
CORS_ALLOW_ORIGINS=https://your-frontend.example.com
```

**API 端口**：默认 `8002`，可在 `server.py` 中修改

**数据存储**：

- 上传的文件保存在 `backend/temp_data/`
- 生成的图片保存在 `backend/static/images/`

### 前端配置

**API 地址**：在 `frontend/src/config/api.ts` 中配置后端地址

```typescript
export const API_BASE_URL = 'http://localhost:8002';
```

## 使用指南

### 1. 上传数据

- 在聊天输入框左侧点击上传按钮选择 CSV 文件
- 上传成功后，数据预览会自动显示在前 10 行
- 系统会自动识别变量类型（离散/连续）

### 2. 分析数据

#### 方式一：可视化分析

- 在数据预览区域点击两个变量
- 系统自动计算相关性并生成合适的图表

#### 方式二：AI 对话分析

- 在左侧对话界面输入问题，例如：
  - "分析一下数据的整体情况"
  - "计算各列的平均值"
  - "按 state 分组统计平均 sales，并给前 10 行"
  - "对 MonthlyCharges 在 churn=yes/no 两组做 t 检验"
  - "检验 contract 与 churn 是否有关（卡方检验）"
  - "绘制年龄和收入的散点图"
- Agent 会执行 Python 代码并返回结果
- 如果生成了图表，会自动显示在可视化面板

### 统计能力（v1）

- 支持的 `stats` helper：
  - `describe_numeric`
  - `describe_categorical`
  - `group_summary`
  - `correlation`
  - `t_test`
  - `chi_square`
  - `anova`
- 所有统计结果以结构化 artifact 返回，包含：
  - `artifact_type`
  - `artifact_id`
  - `created_at`
  - `dataset_id`
  - `warnings`
- 结果类型：
  - `stats_result`
  - `test_result`
  - `schema_profile`
  - `preprocess_result`
  - `model_prep_plan`
  - `model_result`
  - `metrics_result`
  - `feature_importance_result`
- `group_summary` 的 `rate` 支持显式正类参数（`positive_label`），并在结果中返回推断来源与 warning。
- 数据分析路由采用“可解释评分决策”（包含 score/threshold/reasons），默认以确定性规则为主。

### Baseline ML 能力（v2）

- 支持的 `ml` helper：
  - `logistic_fit`
  - `linear_regression_fit`
  - `metrics`
  - `feature_importance`
  - `latest`
- 建模输入预处理（受限、可解释）：
  - 数值特征：中位数填充
  - 分类特征：`Unknown` 填充 + One-Hot（`handle_unknown="ignore"`）
  - `identifier_like` / `text_like` / `datetime_like` 默认排除并给 warning
- 仅在用户明确建模意图时触发 ML 路由（训练/预测/评估/特征重要性）。
- 所有模型结果都以结构化 artifact 返回并可在后续轮次复用。

### 当前限制

- 当前版本是受限 helper 设计，不是通用 notebook 执行环境。
- 不支持任意 `import scipy/sklearn`（模型侧不可直接导入）。
- baseline ML 仅提供逻辑回归与线性回归，不支持随机森林、XGBoost、SHAP、AutoML。
- 不支持 AutoML、超参搜索、任意模型持久化。
- 统计检验结果用于基础分析场景，样本过小会返回 warning，请结合业务判断。
- 系数和特征重要性只用于解释模型相关性，不代表因果结论。

### 3. 查看结果

- **对话结果**：在左侧对话界面查看 AI 的分析结果
- **可视化图表**：图表结果显示在聊天流中的结构化卡片
- **生成图片**：Agent 生成的图片保存在 `backend/static/images/`
