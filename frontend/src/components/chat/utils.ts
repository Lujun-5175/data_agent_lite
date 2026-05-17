import type { ServerUploadResponse, UploadedDataset } from '../../types/data';
import type { ChatMessage, RouteInfo } from './types';

export function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function getViewportHeight() {
  if (typeof window === 'undefined') return 900;
  return window.innerHeight;
}

export function formatCellValue(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function normalizeUploadedDataset(result: ServerUploadResponse, fallbackFilename: string): UploadedDataset {
  const datasetId = (result.dataset_id ?? '').trim();
  if (!datasetId) throw new Error('后端未返回 dataset_id。');
  return {
    datasetId,
    filename: result.original_filename ?? result.filename ?? fallbackFilename,
    preview: Array.isArray(result.preview) ? result.preview : [],
    columns: Array.isArray(result.columns) ? result.columns : [],
    originalRowCount: typeof result.original_row_count === 'number' ? result.original_row_count : 0,
    rowCount: typeof result.row_count === 'number' ? result.row_count : 0,
    columnCount: typeof result.column_count === 'number' ? result.column_count : 0,
    previewCount: typeof result.preview_count === 'number' ? result.preview_count : 0,
    analysisBasis: result.analysis_basis ?? 'working_df',
    preprocessingLog: Array.isArray(result.preprocessing_log) ? result.preprocessing_log : [],
    recommendedPrompts: Array.isArray(result.recommended_prompts)
      ? result.recommended_prompts.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
  };
}

export function extractStreamText(payload: Record<string, unknown>) {
  const content = payload.content ?? payload.text ?? payload.delta;
  return typeof content === 'string' ? content : '';
}

export function extractImageUrl(payload: Record<string, unknown>) {
  const imageUrl = payload.image_url;
  return typeof imageUrl === 'string' && imageUrl.trim() ? imageUrl : '';
}

export function normalizeRouteInfo(payload: Record<string, unknown>): RouteInfo | null {
  const primaryMode = typeof payload.primary_mode === 'string' ? payload.primary_mode : '';
  const confidenceScore = typeof payload.confidence_score === 'number' ? payload.confidence_score : null;
  const intentType = typeof payload.intent_type === 'string' ? payload.intent_type : '';
  const confidence = typeof payload.confidence === 'string' ? payload.confidence : 'medium';
  const routeSource = typeof payload.route_source === 'string' ? payload.route_source : '';
  const finalBranch = typeof payload.final_branch === 'string' ? payload.final_branch : '';
  if (!intentType || !routeSource) return null;

  return {
    primaryMode,
    confidenceScore,
    intentType,
    confidence,
    routeSource,
    conflictFlags: Array.isArray(payload.conflict_flags)
      ? payload.conflict_flags.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    ambiguityFlags: Array.isArray(payload.ambiguity_flags)
      ? payload.ambiguity_flags.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    guardrailActions: Array.isArray(payload.guardrail_actions)
      ? payload.guardrail_actions.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    fallbackReasons: Array.isArray(payload.fallback_reasons)
      ? payload.fallback_reasons.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    suggestedPlan: Array.isArray(payload.suggested_plan)
      ? payload.suggested_plan.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    requestedCapabilities: Array.isArray(payload.requested_capabilities)
      ? payload.requested_capabilities.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    requiresMl: payload.requires_ml === true,
    requiresChart: payload.requires_chart === true,
    requiresPythonAnalysis: payload.requires_python_analysis === true,
    isFollowUp: payload.is_follow_up === true,
    needsDataset: payload.needs_dataset === true,
    needsToolExecution: payload.needs_tool_execution === true,
    needsArtifactContext: payload.needs_artifact_context === true,
    finalBranch,
    taskPlanAvailable: payload.task_plan_available === true,
    taskPlanGoal: typeof payload.task_plan_goal === 'string' ? payload.task_plan_goal : '',
    taskPlanConfidence: typeof payload.task_plan_confidence === 'number' ? payload.task_plan_confidence : null,
    taskPlanTasks: Array.isArray(payload.task_plan_tasks)
      ? payload.task_plan_tasks
          .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
          .map((item) => ({
            taskId: typeof item.task_id === 'string' ? item.task_id : '',
            taskType: typeof item.task_type === 'string' ? item.task_type : '',
            description: typeof item.description === 'string' ? item.description : '',
            dependsOn: Array.isArray(item.depends_on)
              ? item.depends_on.filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)
              : [],
            requiredOutputs: Array.isArray(item.required_outputs)
              ? item.required_outputs.filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)
              : [],
          }))
          .filter((item) => item.taskId || item.taskType || item.description)
      : [],
    taskPlanAmbiguityFlags: Array.isArray(payload.task_plan_ambiguity_flags)
      ? payload.task_plan_ambiguity_flags.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    taskPlanAssumptions: Array.isArray(payload.task_plan_assumptions)
      ? payload.task_plan_assumptions.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [],
    taskPlanAttempted: payload.task_plan_attempted === true,
    taskPlanGenerationFailed: payload.task_plan_generation_failed === true,
  };
}

export function summarizeRouteInfo(routeInfo: RouteInfo) {
  const primaryModeLabelMap: Record<string, string> = {
    direct_answer: '直接回答',
    dataset_overview: '数据概览',
    analysis: '分析',
    visualization: '可视化',
    modeling: '建模',
    artifact_followup: '结果续问',
    mixed: '混合流程',
    clarification: '澄清问题',
  };
  const intentLabelMap: Record<string, string> = {
    analysis: '分析',
    ml: '建模',
    chart: '图表',
    mixed: '混合',
    followup: '续问',
    dataset_overview: '概览',
  };
  const confidenceLabelMap: Record<string, string> = {
    low: '低置信度',
    medium: '中置信度',
    high: '高置信度',
  };
  const routeSourceLabelMap: Record<string, string> = {
    llm_primary: 'LLM 主判定',
    llm_with_guardrail: 'LLM + Guardrail',
    heuristic_fallback: '规则回退',
  };
  const intentLabel = intentLabelMap[routeInfo.intentType] ?? routeInfo.intentType;
  const confidenceLabel = confidenceLabelMap[routeInfo.confidence] ?? routeInfo.confidence;
  const routeSourceLabel = routeSourceLabelMap[routeInfo.routeSource] ?? routeInfo.routeSource;
  const primaryModeLabel = primaryModeLabelMap[routeInfo.primaryMode] ?? (routeInfo.primaryMode || intentLabel);
  return `${primaryModeLabel} · ${confidenceLabel} · ${routeSourceLabel}`;
}

export function buildMessageHistory(messages: ChatMessage[]) {
  return messages
    .filter((message) => message.kind === 'text' && message.content.trim() !== '')
    .slice(-12)
    .map((message) => ({
      type: message.type === 'user' ? 'human' : 'ai',
      content: message.content,
    }));
}

export function parseSseBlocks(buffer: string) {
  const blocks = buffer.split('\n\n');
  if (!buffer.endsWith('\n\n')) {
    return { blocks: blocks.slice(0, -1), remainder: blocks.at(-1) ?? '' };
  }
  return { blocks, remainder: '' };
}

export function parseSseEventBlock(block: string) {
  const lines = block.split('\n');
  const eventLine = lines.find((line) => line.startsWith('event:'));
  const dataLine = lines.find((line) => line.startsWith('data:'));
  if (!eventLine || !dataLine) return null;

  const eventType = eventLine.slice(6).trim();
  const dataContent = dataLine.slice(5).trim();
  if (!dataContent) return null;

  try {
    return {
      eventType,
      payload: JSON.parse(dataContent) as Record<string, unknown>,
    };
  } catch {
    return null;
  }
}
