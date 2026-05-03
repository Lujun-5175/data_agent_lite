import type { ServerUploadResponse, UploadedDataset } from '../../types/data';
import type { ChatMessage } from './types';

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

export function buildMessageHistory(messages: ChatMessage[]) {
  return messages
    .filter((message) => message.kind === 'text' && message.content.trim() !== '')
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
