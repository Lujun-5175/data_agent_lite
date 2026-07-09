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
  if (!datasetId) throw new Error('Backend did not return a dataset_id.');
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
  const routeSource = typeof payload.route_source === 'string' ? payload.route_source : 'llm';
  if (!primaryMode) return null;

  const intentType = typeof payload.intent_type === 'string' ? payload.intent_type : primaryMode;
  const finalBranch = typeof payload.final_branch === 'string' ? payload.final_branch : primaryMode;

  return {
    primaryMode,
    confidenceScore: null,
    intentType,
    confidence: 'medium',
    routeSource,
    conflictFlags: [],
    ambiguityFlags: [],
    guardrailActions: [],
    fallbackReasons: [],
    suggestedPlan: [],
    requestedCapabilities: [],
    requiresMl: false,
    requiresChart: false,
    requiresPythonAnalysis: false,
    isFollowUp: false,
    needsDataset: payload.needs_dataset === true,
    needsToolExecution: payload.needs_tool_execution === true,
    needsArtifactContext: false,
    finalBranch,
    taskPlanAvailable: false,
    taskPlanGoal: '',
    taskPlanConfidence: null,
    taskPlanTasks: [],
    taskPlanAmbiguityFlags: [],
    taskPlanAssumptions: [],
    taskPlanAttempted: false,
    taskPlanGenerationFailed: false,
  };
}

export function summarizeRouteInfo(routeInfo: RouteInfo) {
  const primaryModeLabelMap: Record<string, string> = {
    direct_answer: 'Direct Answer',
    dataset_overview: 'Dataset Overview',
    analysis: 'Analysis',
    visualization: 'Visualization',
    modeling: 'Modeling',
    artifact_followup: 'Follow-up',
    mixed: 'Mixed',
    clarification: 'Clarification',
  };
  const intentLabelMap: Record<string, string> = {
    analysis: 'Analysis',
    ml: 'Modeling',
    chart: 'Chart',
    mixed: 'Mixed',
    followup: 'Follow-up',
    dataset_overview: 'Overview',
  };
  const confidenceLabelMap: Record<string, string> = {
    low: 'Low Confidence',
    medium: 'Medium Confidence',
    high: 'High Confidence',
  };
  const routeSourceLabelMap: Record<string, string> = {
    llm_primary: 'LLM Primary',
    llm_with_guardrail: 'LLM + Guardrail',
    heuristic_fallback: 'Heuristic Fallback',
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
