from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.data_manager import get_data_info, get_dataset
from src.routing_rules import RoutingContext, decide_dataset_required, decide_stats_intent, interpret_request
from src.tools import (
    fig_inter,
    ml_execute,
    python_inter,
    set_current_dataset_id,
)

logger = logging.getLogger(__name__)
load_dotenv(override=True)


class AgentContext(BaseModel):
    dataset_id: str | None = Field(default=None, description="当前分析的数据集 ID")


def _extract_dataset_id_from_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, AgentContext):
        if value.dataset_id and value.dataset_id.strip():
            return value.dataset_id.strip()
        return None
    if isinstance(value, dict):
        direct_dataset_id = value.get("dataset_id")
        if isinstance(direct_dataset_id, str) and direct_dataset_id.strip():
            return direct_dataset_id.strip()
        configurable = value.get("configurable")
        if isinstance(configurable, dict):
            nested_dataset_id = configurable.get("dataset_id")
            if isinstance(nested_dataset_id, str) and nested_dataset_id.strip():
                return nested_dataset_id.strip()
        return None
    direct_dataset_id = getattr(value, "dataset_id", None)
    if isinstance(direct_dataset_id, str) and direct_dataset_id.strip():
        return direct_dataset_id.strip()
    configurable = getattr(value, "configurable", None)
    if isinstance(configurable, dict):
        nested_dataset_id = configurable.get("dataset_id")
        if isinstance(nested_dataset_id, str) and nested_dataset_id.strip():
            return nested_dataset_id.strip()
    return None


def _extract_dataset_id(request) -> str | None:
    runtime = getattr(request, "runtime", None)
    if runtime is not None:
        context = getattr(runtime, "context", None)
        dataset_id = _extract_dataset_id_from_value(context)
        if dataset_id:
            logger.debug(
                "dataset_id resolved from runtime.context",
                extra={"dataset_id": dataset_id},
            )
            return dataset_id

        runtime_config = getattr(runtime, "config", None)
        dataset_id = _extract_dataset_id_from_value(runtime_config)
        if dataset_id:
            logger.debug(
                "dataset_id resolved from runtime.config",
                extra={"dataset_id": dataset_id},
            )
            return dataset_id

    request_config = getattr(request, "config", None)
    dataset_id = _extract_dataset_id_from_value(request_config)
    if dataset_id:
        logger.debug(
            "dataset_id resolved from request.config",
            extra={"dataset_id": dataset_id},
        )
        return dataset_id
    return None


model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
)

tools = [
    python_inter,
    fig_inter,
    ml_execute,
]

def _normalize_text(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def get_dataset_required_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_dataset_required(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_dataset_required(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_dataset_required_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def get_stats_intent_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_stats_intent(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_stats_intent(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_stats_intent_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def get_ml_intent_decision(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
):
    return decide_ml_intent(
        RoutingContext(
            message=message,
            dataset_columns=dataset_columns or [],
            prior_analysis_active=prior_analysis_active,
        )
    )


def is_ml_intent(
    message: str,
    *,
    dataset_columns: list[str] | None = None,
    prior_analysis_active: bool = False,
) -> bool:
    decision = get_ml_intent_decision(
        message,
        dataset_columns=dataset_columns,
        prior_analysis_active=prior_analysis_active,
    )
    return decision.matched


def _build_general_chat_messages(messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = [
        SystemMessage(
            content=(
                "你是 Data Agent 的中文助手，目标是“准确、简洁、可执行”。"
                "请默认使用中文，语气专业友好，不要编造不存在的数据、接口或结论。"
                "如果用户问题属于普通聊天、概念解释、学习建议或通识问答，请直接回答。"
                "如果用户明确要求“基于已上传数据”的分析，但当前没有可用数据集，请明确提示先上传 CSV 文件。"
                "当信息不足以得出结论时，先说清缺失信息，再给最小可行下一步。"
            )
        )
    ]

    for message in messages:
        role = message.get("type")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role in {"human", "user"}:
            converted.append(HumanMessage(content=content))
        elif role in {"ai", "assistant"}:
            converted.append(AIMessage(content=content))

    return converted


def _extract_message_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_content = getattr(item, "text", None)
            if isinstance(item_content, str):
                parts.append(item_content)
            else:
                value = getattr(item, "content", None)
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return str(content)


async def generate_general_chat_reply(messages: list[dict[str, Any]]) -> str:
    response = await model.ainvoke(_build_general_chat_messages(messages))
    return _extract_message_text(response)


@dynamic_prompt
def dataset_context_middleware(request) -> str:
    dataset_id = _extract_dataset_id(request)
    set_current_dataset_id(dataset_id)

    runtime = getattr(request, "runtime", None)
    latest_user_message = ""
    if runtime is not None:
        runtime_input = getattr(runtime, "input", None)
        if isinstance(runtime_input, dict):
            raw_messages = runtime_input.get("messages")
            if isinstance(raw_messages, list):
                for msg in reversed(raw_messages):
                    if isinstance(msg, dict) and msg.get("type") in {"human", "user"}:
                        content = msg.get("content")
                        if isinstance(content, str):
                            latest_user_message = content
                            break

    dataset_columns: list[str] = []
    if dataset_id:
        dataset = get_dataset(dataset_id)
        dataset_columns = [column["name"] for column in dataset.columns if "name" in column]
        data_context = get_data_info(dataset_id)
        dataset_scope = f"当前数据集 dataset_id: {dataset_id}，分析基于 {dataset.analysis_basis}。"
    else:
        data_context = "当前未选择数据集。普通聊天可以继续进行；如果用户需要分析具体数据，请先上传 CSV 文件。"
        dataset_scope = "当前没有可用数据集。普通聊天可直接回答，数据分析需先上传 CSV。"

    interpretation = interpret_request(
        RoutingContext(
            message=latest_user_message,
            dataset_columns=dataset_columns,
            prior_analysis_active=bool(dataset_id),
        ),
        use_llm=False,
    )
    stats_decision = get_stats_intent_decision(
        latest_user_message,
        dataset_columns=dataset_columns,
        prior_analysis_active=bool(dataset_id),
    )
    logger.debug("stats intent decision: %s", stats_decision.to_dict())
    logger.debug("request interpretation: %s", interpretation.to_dict())
    if interpretation.intent_type == "ml":
        route_hint = (
            "这是明确建模请求。选择最小必要步骤，"
            "只有在确实需要训练、评估或特征重要性时才调用 `ml_execute`。"
        )
    elif interpretation.intent_type == "mixed":
        route_hint = (
            "这是混合工作流。先执行 analysis 部分，再判断是否需要 `ml_execute`。"
            "不要把探索性分析直接升级成建模。"
        )
    elif interpretation.intent_type == "chart":
        route_hint = "这是绘图请求。先做最小必要分析，再使用 `fig_inter` 生成图表。"
    elif interpretation.intent_type == "followup":
        route_hint = "这是跟进/续问请求。优先复用最近结构化结果，必要时再补充分析或 ML。"
    elif stats_decision.matched:
        route_hint = f"检测到统计意图（stats score={stats_decision.score:.2f}），优先使用 stats.* helper。"
    else:
        route_hint = (
            "先根据 interpretation 选择最小必要工具。"
            "统计问题优先 stats.*，探索性分析优先 python_inter 或 stats.*，"
            "明确建模请求才使用 `ml_execute`，绘图需求使用 `fig_inter`。"
        )

    return f"""你是 Data Agent 的高级数据分析助手。你的首要目标是：结果正确、过程可复核、表达清晰。

【当前数据集状态】
{dataset_scope}

【数据集摘要】
{data_context}

【你的职责】
1. 对“基于数据的问题”优先调用工具：数据理解/预处理问题优先 `profile.*`，统计分析优先 `stats.*`，明确建模请求再用 `ml_execute`，图表需求用 `fig_inter`。
2. 变量 `df` 是只读数据视图；`data`、`viz`、`stats`、`profile`、`ml` 是白名单 helper API。优先用 helper API，不要依赖未声明能力。
   对于明确建模请求，先判断是否确实需要训练/评估模型再决定是否调用 `ml_execute`；不要把分析性请求误判成建模请求。
   对于探索性分析、过滤、聚合、比较、概览，请优先使用 `python_inter` 或 `stats.*`，必要时再补 `ml_execute`。
3. 只能基于当前数据集和工具输出作答；禁止臆测缺失数据、禁止虚构计算结果。
4. 分析时先完成计算再回答，不要只给思路。若结果为空或样本不足，要明确说明并给出可执行下一步。
5. 绘图任务要给出简短结论（图展示了什么）并保持标题/轴含义清晰。
6. 如果用户请求涉及不存在列、无效筛选或不受支持操作，请直接说明原因并给替代方案。
7. 若用户询问“字段语义、可建模列、预处理步骤”，优先返回结构化 artifact 结果（如 schema_profile / preprocess_result / model_prep_plan）再给简要总结。
8. baseline ML 仅支持逻辑回归/线性回归；不支持 AutoML、随机森林、XGBoost、SHAP。遇到越界请求请明确拒绝并给可执行替代。
9. 若用户显式要求模型指标或特征重要性，应按需继续调用 `ml_execute`，并分别使用 `action="metrics"` / `action="feature_importance"`，复用最近的模型 artifact。

【输出风格】
- 默认中文，先给结论，再给关键依据（核心数字/分组结果/趋势）。
- 避免冗长，不重复用户问题，不输出无意义模板话术。
- 若用户要求 Top N、筛选、分组、时间聚合，必须在回答中体现这些约束是否已正确执行。

【多轮上下文规则】
- 继承当前会话中的筛选范围与目标指标（如“现在只看 California”）。
- 若跟进问题存在歧义，先用一句话确认你将沿用的口径，再给结果。

【当前路由提示】
- {route_hint}

【无数据集时的规则】
- 若用户是普通聊天：直接回答。
- 若用户明确要做数据分析：提示先上传 CSV 文件后再分析。
"""


graph = create_agent(
    model=model,
    tools=tools,
    middleware=[dataset_context_middleware],
    context_schema=AgentContext,
)
