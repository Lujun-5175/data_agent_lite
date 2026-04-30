import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot,
  BarChart3,
  FileSpreadsheet,
  GripVertical,
  Image as ImageIcon,
  Paperclip,
  RefreshCcw,
  Send,
  Table2,
  Trash2,
  TriangleAlert,
  UserRound,
} from 'lucide-react';
import { toast } from 'sonner';
import { API_ENDPOINTS, getFriendlyErrorMessage, readApiErrorInfo } from '../config/api';
import { CHAT_SHELL_CLASS } from '../config/layout';
import { SAMPLE_DATASETS, type SampleDataset } from '../config/sampleDatasets';
import type { ServerUploadResponse, UploadedDataset } from '../types/data';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';

type MessageKind = 'text' | 'status' | 'error' | 'image' | 'dataset_card';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  kind: MessageKind;
  imageUrl?: string;
  filename?: string;
  datasetPayload?: UploadedDataset;
}

const initialMessages: Message[] = [
  {
    id: '1',
    type: 'assistant',
    kind: 'text',
    content: '你好，欢迎来到 Data Agent。你可以直接聊天，也可以在输入框左侧上传 CSV，让我帮你分析、预览和可视化。',
    timestamp: new Date(),
  },
];

const PREVIEW_MIN_HEIGHT = 240;
const PREVIEW_DEFAULT_HEIGHT = 336;
const PREVIEW_MAX_RATIO = 0.58;
const PREVIEW_MAX_ABS = 620;

interface ChatInterfaceProps {
  clearTrigger: number;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getViewportHeight() {
  if (typeof window === 'undefined') return 900;
  return window.innerHeight;
}

function formatCellValue(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeUploadedDataset(result: ServerUploadResponse, fallbackFilename: string): UploadedDataset {
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

function extractStreamText(payload: Record<string, unknown>) {
  const content = payload.content ?? payload.text ?? payload.delta;
  return typeof content === 'string' ? content : '';
}

function extractImageUrl(payload: Record<string, unknown>) {
  const imageUrl = payload.image_url;
  return typeof imageUrl === 'string' && imageUrl.trim() ? imageUrl : '';
}

function useViewportHeight() {
  const [height, setHeight] = useState(() => getViewportHeight());
  useEffect(() => {
    const handleResize = () => setHeight(getViewportHeight());
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return height;
}

function ResizablePreviewPanel({
  columns,
  rows,
  datasetId,
  height,
  onHeightChange,
}: {
  columns: UploadedDataset['columns'];
  rows: UploadedDataset['preview'];
  datasetId: string;
  height: number;
  onHeightChange: (nextHeight: number) => void;
}) {
  const viewportHeight = useViewportHeight();
  const maxHeight = useMemo(
    () => clamp(Math.round(viewportHeight * PREVIEW_MAX_RATIO), PREVIEW_MIN_HEIGHT, PREVIEW_MAX_ABS),
    [viewportHeight]
  );
  const dragStateRef = useRef<{ startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    onHeightChange(clamp(height, PREVIEW_MIN_HEIGHT, maxHeight));
  }, [height, maxHeight, onHeightChange]);

  const handlePointerDown = (event: import('react').PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    dragStateRef.current = { startY: event.clientY, startHeight: height };

    const handleMove = (moveEvent: PointerEvent) => {
      if (!dragStateRef.current) return;
      const delta = moveEvent.clientY - dragStateRef.current.startY;
      onHeightChange(clamp(dragStateRef.current.startHeight + delta, PREVIEW_MIN_HEIGHT, maxHeight));
    };

    const handleUp = () => {
      dragStateRef.current = null;
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  return (
    <div
      className="mt-3 overflow-hidden rounded-[16px] border border-slate-200 bg-white shadow-[0_8px_24px_rgba(15,23,42,0.04)]"
      style={{ height, maxHeight }}
    >
      <div className="flex h-full flex-col">
        <div className="shrink-0 border-b border-slate-200 bg-slate-50/80 px-4 py-2.5">
          <div className="flex items-center justify-between gap-3 text-[11px] font-medium text-slate-500">
            <span>数据预览</span>
            <span>{rows.length > 0 ? `显示 ${rows.length} 行` : '暂无预览行'}</span>
          </div>
          <p className="mt-1 text-[11px] leading-5 text-slate-500">
            这个容器只负责表格内部滚动，拖拽下面的把手可以调整高度，不会把聊天区整体撑开。
          </p>
        </div>

        <div className="min-h-0 flex-1 overflow-auto">
          <table className="w-full min-w-max border-separate border-spacing-0">
            <thead className="sticky top-0 z-10 bg-white">
              <tr>
                {columns.map((column) => (
                  <th
                    key={`${datasetId}-${column.name}`}
                    className="border-b border-slate-200 px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-600"
                  >
                    <div className="flex items-center gap-2">
                      <span className="truncate">{column.name}</span>
                      <span className="rounded-full border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                        {column.type === 'numerical' ? '数值' : '分类'}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`${datasetId}-${rowIndex}`} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                  {columns.map((column) => (
                    <td
                      key={`${datasetId}-${column.name}-${rowIndex}`}
                      className="border-b border-slate-100 px-4 py-3 text-xs font-medium text-slate-800"
                    >
                      {formatCellValue(row[column.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <button
          type="button"
          onPointerDown={handlePointerDown}
          className="flex h-5 shrink-0 items-center justify-center border-t border-slate-200 bg-slate-50 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 touch-none"
          aria-label="拖拽调整预览高度"
        >
          <GripVertical className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

function SampleDataPanel({
  samples,
  loadingSampleId,
  disabled,
  onSelect,
}: {
  samples: SampleDataset[];
  loadingSampleId: string | null;
  disabled: boolean;
  onSelect: (sample: SampleDataset) => void;
}) {
  return (
    <section className="rounded-[24px] border border-slate-200/80 bg-white/82 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.045)] backdrop-blur-sm md:p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            <BarChart3 className="h-3.5 w-3.5" />
            示例数据一键试用
          </div>
          <h2 className="mt-3 text-lg font-semibold tracking-tight text-slate-950 md:text-xl">
            选择一个示例数据集，立即开始体验
          </h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600">
            不需要准备 CSV，点击加载后即可测试数据预览、统计分析、图表生成和建模流程。
          </p>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        {samples.map((sample) => {
          const isLoading = loadingSampleId === sample.id;
          return (
            <article
              key={sample.id}
              className="flex min-h-[210px] flex-col rounded-[20px] border border-slate-200 bg-white p-4 shadow-[0_8px_22px_rgba(15,23,42,0.035)]"
            >
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] border border-slate-200 bg-slate-50 text-slate-700">
                  <FileSpreadsheet className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <h3 className="text-base font-semibold text-slate-950">{sample.name}</h3>
                  <p className="mt-1 text-xs leading-5 text-slate-600">{sample.description}</p>
                </div>
              </div>
              <div className="mt-4 rounded-[14px] border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600">
                {sample.columnCount} 列 x {sample.rowCount.toLocaleString()} 行
              </div>
              <button
                type="button"
                onClick={() => onSelect(sample)}
                disabled={disabled || isLoading}
                className="mt-4 flex min-h-11 w-full items-center justify-center rounded-[14px] border border-slate-300 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-950 shadow-[0_6px_16px_rgba(15,23,42,0.08)] transition-colors hover:border-slate-900 hover:bg-white disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-500"
              >
                {isLoading ? '加载中...' : '加载'}
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SuggestedPrompts({
  prompts,
  disabled,
  onSelect,
}: {
  prompts: string[];
  disabled: boolean;
  onSelect: (prompt: string) => void;
}) {
  if (prompts.length === 0) return null;

  return (
    <div className="rounded-[18px] border border-slate-200/80 bg-white/78 p-3 shadow-[0_8px_20px_rgba(15,23,42,0.035)]">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">推荐问题</div>
      <div className="flex flex-wrap gap-2">
        {prompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onSelect(prompt)}
            disabled={disabled}
            className="min-h-11 max-w-full rounded-[14px] border border-slate-200 bg-white px-3 py-2 text-left text-xs font-medium leading-5 text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

function ChatBubble({ message }: { message: Message }) {
  const isUser = message.type === 'user';
  const isStatus = message.kind === 'status';
  const isError = message.kind === 'error';
  const bubbleClass = isUser
    ? 'ml-auto max-w-[72%] border-sky-100 bg-sky-50 text-slate-900 shadow-[0_8px_24px_rgba(15,23,42,0.04)]'
    : isStatus
      ? 'max-w-[72%] border-amber-100 bg-amber-50/80 text-slate-800'
      : isError
        ? 'max-w-[72%] border-rose-100 bg-rose-50/80 text-slate-800'
        : 'max-w-[72%] border-slate-200 bg-white text-slate-900';

  return (
    <div className={`flex w-full items-start gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div
          className={`mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 shadow-[0_4px_12px_rgba(15,23,42,0.04)] ${
            isError ? 'bg-rose-500' : 'bg-slate-900'
          }`}
        >
          {isError ? <TriangleAlert className="h-4 w-4 text-white" /> : <Bot className="h-4 w-4 text-white" />}
        </div>
      )}

      <div className={`flex min-h-[48px] items-center rounded-[22px] border px-4 py-3 text-[15px] leading-7 shadow-[0_8px_20px_rgba(15,23,42,0.035)] ${bubbleClass}`}>
        <div className="whitespace-pre-wrap break-words text-[15px] font-medium leading-7 text-slate-900">
          {message.content}
        </div>
      </div>

      {isUser && (
        <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-slate-50 shadow-[0_4px_12px_rgba(15,23,42,0.04)]">
          <UserRound className="h-4 w-4 text-slate-700" />
        </div>
      )}
    </div>
  );
}

function ImageCard({ message }: { message: Message }) {
  return (
    <div className="flex w-full items-start gap-3">
      <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-white shadow-[0_4px_12px_rgba(15,23,42,0.04)]">
        <ImageIcon className="h-4 w-4 text-slate-700" />
      </div>
      <div className="w-full max-w-[860px] rounded-[22px] border border-slate-200/80 bg-white p-4 shadow-[0_8px_22px_rgba(15,23,42,0.04)]">
        <div className="mb-3 text-sm font-medium text-slate-700">{message.content}</div>
        {message.imageUrl && (
          <img
            src={message.imageUrl}
            alt={message.filename || '生成的图表'}
            className="max-h-[420px] w-full rounded-[14px] border border-slate-200 object-contain"
          />
        )}
      </div>
    </div>
  );
}

export function ChatInterface({ clearTrigger }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<UploadedDataset | null>(null);
  const [suggestedPrompts, setSuggestedPrompts] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const assistantMessageIdRef = useRef<string | null>(null);
  const statusMessageIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (clearTrigger <= 0) return;
    setMessages(initialMessages);
    setInputValue('');
    setIsLoading(false);
    setIsUploading(false);
    setLoadingSampleId(null);
    setIsDeleting(false);
    setUploadedDataset(null);
    setSuggestedPrompts([]);
    assistantMessageIdRef.current = null;
    statusMessageIdRef.current = null;
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [clearTrigger]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isLoading, uploadedDataset]);

  const buildMessageHistory = () =>
    messages
      .filter((msg) => msg.kind === 'text' && msg.content.trim() !== '')
      .map((msg) => ({
        type: msg.type === 'user' ? 'human' : 'ai',
        content: msg.content,
      }));

  const handleFileUploadClick = () => fileInputRef.current?.click();

  const handleFileSelect = async (file: File, options?: { sample?: SampleDataset }) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      const message = '请上传 CSV 格式的文件';
      toast.error(message);
      setMessages((prev) => [
        ...prev,
        { id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() },
      ]);
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(API_ENDPOINTS.UPLOAD, { method: 'POST', body: formData });
      if (!response.ok) {
        const { code, message } = await readApiErrorInfo(response, '文件上传失败，请检查后端服务是否正常运行');
        throw new Error(getFriendlyErrorMessage(code, message));
      }

      const result = (await response.json()) as ServerUploadResponse;
      const dataset = normalizeUploadedDataset(result, file.name);
      setUploadedDataset(dataset);
      setSuggestedPrompts(options?.sample?.prompts ?? []);
      setMessages((prev) => [
        ...prev.filter((msg) => msg.kind !== 'dataset_card'),
        {
          id: `${dataset.datasetId}-dataset-card`,
          type: 'assistant',
          kind: 'dataset_card',
          content: '',
          datasetPayload: dataset,
          timestamp: new Date(),
        },
      ]);
      toast.success(
        options?.sample
          ? `示例数据已加载：${options.sample.name}`
          : result.message || `成功加载文件【${dataset.filename}】`
      );
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (error) {
      const message = error instanceof Error ? error.message : '文件上传失败，请检查后端服务是否正常运行';
      setMessages((prev) => [
        ...prev,
        { id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() },
      ]);
      toast.error(message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSampleDatasetSelect = async (sample: SampleDataset) => {
    if (isUploading || isLoading || loadingSampleId) return;

    setLoadingSampleId(sample.id);
    try {
      const response = await fetch(sample.path);
      if (!response.ok) {
        throw new Error(`示例数据加载失败：${sample.filename}`);
      }
      const csvBlob = await response.blob();
      const csvFile = new File([csvBlob], sample.filename, { type: 'text/csv' });
      await handleFileSelect(csvFile, { sample });
    } catch (error) {
      const message = error instanceof Error ? error.message : '示例数据加载失败';
      setMessages((prev) => [
        ...prev,
        { id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() },
      ]);
      toast.error(message);
    } finally {
      setLoadingSampleId(null);
    }
  };

  const handleDeleteDataset = async (targetDatasetId: string) => {
    if (!targetDatasetId) return;
    setIsDeleting(true);
    try {
      const response = await fetch(API_ENDPOINTS.DELETE_DATASET(targetDatasetId), { method: 'DELETE' });
      if (!response.ok) {
        const { code, message } = await readApiErrorInfo(response, '删除数据集失败，请稍后重试');
        throw new Error(getFriendlyErrorMessage(code, message));
      }
      setUploadedDataset(null);
      setSuggestedPrompts([]);
      setMessages((prev) => prev.filter((msg) => msg.kind !== 'dataset_card'));
      setMessages((prev) => [
        ...prev,
        {
          id: `status-${Date.now()}`,
          type: 'assistant',
          kind: 'status',
          content: '当前数据集已移除，你可以继续聊天或重新上传文件。',
          timestamp: new Date(),
        },
      ]);
      toast.success('数据集已删除');
    } catch (error) {
      const message = error instanceof Error ? error.message : '删除数据集失败，请稍后重试';
      setMessages((prev) => [
        ...prev,
        { id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() },
      ]);
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  const upsertAssistantMessage = (assistantMessageId: string, content: string) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === assistantMessageId ? { ...msg, content, kind: 'text' as const } : msg))
    );
  };

  const upsertStatusMessage = (assistantMessageId: string, content: string) => {
    if (!statusMessageIdRef.current) {
      const statusMessageId = `${assistantMessageId}-status`;
      statusMessageIdRef.current = statusMessageId;
      setMessages((prev) => [
        ...prev,
        { id: statusMessageId, type: 'assistant', kind: 'status', content, timestamp: new Date() },
      ]);
      return;
    }

    setMessages((prev) =>
      prev.map((msg) => (msg.id === statusMessageIdRef.current ? { ...msg, content, kind: 'status' as const } : msg))
    );
  };

  const appendImageMessage = (assistantMessageId: string, imageUrl: string, filename?: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: `${assistantMessageId}-image-${Date.now()}`,
        type: 'assistant',
        kind: 'image',
        content: filename ? `图表结果：${filename}` : '图表结果',
        imageUrl,
        filename,
        timestamp: new Date(),
      },
    ]);
  };

  const handleSendMessage = async (messageOverride?: string) => {
    const outgoingContent = (messageOverride ?? inputValue).trim();
    if (!outgoingContent || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: outgoingContent,
      timestamp: new Date(),
      kind: 'text',
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();
    assistantMessageIdRef.current = assistantMessageId;
    statusMessageIdRef.current = null;
    setMessages((prev) => [
      ...prev,
      { id: assistantMessageId, type: 'assistant', content: '', timestamp: new Date(), kind: 'text' },
    ]);

    let accumulatedContent = '';
    try {
      const datasetId = uploadedDataset?.datasetId ?? null;
      const requestBody: Record<string, unknown> = {
        ...(datasetId ? { dataset_id: datasetId } : {}),
        input: { messages: [...buildMessageHistory(), { type: 'human', content: userMessage.content }] },
        config: { configurable: datasetId ? { dataset_id: datasetId } : {} },
      };

      const response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        const { code, message } = await readApiErrorInfo(response, '调用AI助手失败');
        throw new Error(getFriendlyErrorMessage(code, message));
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finished = false;

      if (reader) {
        while (!finished) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split('\n\n');
          if (!buffer.endsWith('\n\n')) {
            buffer = blocks.pop() || '';
          } else {
            buffer = '';
          }

          for (const block of blocks) {
            if (!block.trim()) continue;
            const lines = block.split('\n');
            const eventLine = lines.find((line) => line.startsWith('event:'));
            const dataLine = lines.find((line) => line.startsWith('data:'));
            if (!eventLine || !dataLine) continue;

            const eventType = eventLine.slice(6).trim();
            const dataContent = dataLine.slice(5).trim();
            if (!dataContent) continue;

            let parsed: Record<string, unknown> | null = null;
            try {
              parsed = JSON.parse(dataContent) as Record<string, unknown>;
            } catch {
              continue;
            }

            if (eventType === 'message_chunk') {
              const chunkText = extractStreamText(parsed);
              if (chunkText) {
                accumulatedContent += chunkText;
                upsertAssistantMessage(assistantMessageId, accumulatedContent);
              }
              continue;
            }

            if (eventType === 'tool_start') {
              upsertStatusMessage(assistantMessageId, '正在生成分析结果…');
              continue;
            }

            if (eventType === 'tool_end') {
              upsertStatusMessage(assistantMessageId, '分析已完成');
              continue;
            }

            if (eventType === 'image_generated') {
              const imageUrl = extractImageUrl(parsed);
              if (imageUrl) {
                appendImageMessage(assistantMessageId, imageUrl, isNonEmptyString(parsed.filename) ? parsed.filename : undefined);
              }
              upsertStatusMessage(assistantMessageId, '图表结果已附在当前对话中');
              continue;
            }

            if (eventType === 'error') {
              const errorMessage = getFriendlyErrorMessage(
                isNonEmptyString(parsed.code) ? parsed.code : undefined,
                isNonEmptyString(parsed.message) ? parsed.message : '调用AI助手失败'
              );
              throw new Error(errorMessage);
            }

            if (eventType === 'done') {
              finished = true;
              break;
            }
          }
        }
      }
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message === 'Failed to fetch'
            ? '后端服务连接失败，请确认 8002 端口已启动。'
            : error.message
          : '调用AI助手时出现错误';

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId ? { ...msg, content: `抱歉，${message}`, kind: 'error' as const } : msg
        )
      );
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSendMessage();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void handleFileSelect(file);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileInputChange} className="hidden" />

      <div className="min-h-0 flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto custom-scrollbar py-4 md:py-5">
          <div className={`${CHAT_SHELL_CLASS} pb-5 pt-1 md:pb-6 md:pt-2`}>
            <div className="flex w-full flex-col gap-4">
              {!uploadedDataset && (
                <SampleDataPanel
                  samples={SAMPLE_DATASETS}
                  loadingSampleId={loadingSampleId}
                  disabled={isUploading || isLoading || loadingSampleId !== null}
                  onSelect={(sample) => void handleSampleDatasetSelect(sample)}
                />
              )}

              <SuggestedPrompts
                prompts={suggestedPrompts}
                disabled={isLoading}
                onSelect={(prompt) => void handleSendMessage(prompt)}
              />

              {messages.map((message) => {
                if (message.kind === 'image') {
                  return <ImageCard key={message.id} message={message} />;
                }

                if (message.kind === 'dataset_card' && message.datasetPayload) {
                  const isActive = uploadedDataset?.datasetId === message.datasetPayload.datasetId;
                  return (
                    <div key={message.id} className="flex w-full items-start gap-3">
                      <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-white shadow-[0_4px_12px_rgba(15,23,42,0.035)]">
                        <Table2 className="h-4 w-4 text-slate-700" />
                      </div>
                      <DataPreviewCard
                        payload={message.datasetPayload}
                        isActive={isActive}
                        onReplace={handleFileUploadClick}
                        onDelete={handleDeleteDataset}
                        isUploading={isUploading}
                        isDeleting={isDeleting}
                      />
                    </div>
                  );
                }

                return <ChatBubble key={message.id} message={message} />;
              })}

              {isLoading && (
                <div className="flex w-full items-start gap-3">
                  <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-white shadow-[0_4px_12px_rgba(15,23,42,0.035)]">
                    <Bot className="h-4 w-4 text-slate-700" />
                  </div>
                  <div className="rounded-[18px] border border-slate-200/80 bg-white px-4 py-3 shadow-[0_8px_20px_rgba(15,23,42,0.035)]">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-400/85" />
                      <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-300/85 [animation-delay:0.12s]" />
                      <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-200/85 [animation-delay:0.24s]" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      </div>

      <div className="shrink-0 border-t border-slate-200/70 bg-white/88 backdrop-blur-xl shadow-[0_-8px_24px_rgba(15,23,42,0.04)]">
        <div className={`${CHAT_SHELL_CLASS} space-y-3 py-3 md:py-4`}>
          <div className="flex flex-wrap items-center gap-2 rounded-[16px] border border-slate-200/60 bg-white/70 px-3 py-2 text-xs leading-none text-slate-600 shadow-[0_4px_16px_rgba(15,23,42,0.03)]">
            <span className="inline-flex h-7 items-center rounded-full border border-slate-200 bg-white px-2.5 font-medium text-slate-700">
              当前模式：单栏聊天
            </span>
            {uploadedDataset ? (
              <>
                <span className="inline-flex h-7 items-center rounded-full border border-slate-200 bg-white px-2.5 font-medium text-slate-700">
                  当前数据：{uploadedDataset.filename}
                </span>
                <span>分析基于 working_df，预处理日志已记录。</span>
              </>
            ) : (
              <span>当前未接入数据集，可直接聊天或上传 CSV 开始分析。</span>
            )}
          </div>

          <div className="rounded-[24px] border border-slate-200/60 bg-white/90 p-2.5 shadow-[0_14px_30px_rgba(15,23,42,0.045)] backdrop-blur-sm">
            <div className="flex items-end gap-2.5">
              <Button
                type="button"
                onClick={handleFileUploadClick}
                disabled={isUploading}
                variant="outline"
                className="h-11 w-11 shrink-0 rounded-[16px] border-slate-200/70 bg-white px-0 text-slate-700 hover:bg-slate-50"
              >
                <Paperclip className="h-4 w-4" />
              </Button>

              <div className="flex min-h-[56px] flex-1 items-center rounded-[18px] border border-slate-200/70 bg-white px-1.5 shadow-inner shadow-slate-100/50">
                <Textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入你的问题，或者先上传 CSV 文件..."
                  className="min-h-[52px] max-h-[160px] flex-1 resize-none rounded-[16px] border-0 bg-transparent px-3 py-3 text-[15px] font-medium text-slate-900 shadow-none placeholder:text-slate-400 focus-visible:ring-0 focus-visible:ring-offset-0"
                  disabled={isLoading}
                />
              </div>

              <Button
                type="button"
                onClick={() => void handleSendMessage()}
                disabled={!inputValue.trim() || isLoading}
                className="h-11 w-11 shrink-0 rounded-[16px] border border-slate-900 bg-slate-900 px-0 text-white shadow-[0_4px_12px_rgba(15,23,42,0.10)] hover:bg-slate-800"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DataPreviewCard({
  payload,
  isActive,
  onReplace,
  onDelete,
  isUploading,
  isDeleting,
}: {
  payload: UploadedDataset;
  isActive: boolean;
  onReplace: () => void;
  onDelete: (datasetId: string) => void;
  isUploading: boolean;
  isDeleting: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [panelHeight, setPanelHeight] = useState(PREVIEW_DEFAULT_HEIGHT);
  const viewportHeight = useViewportHeight();
  const panelMaxHeight = useMemo(
    () => clamp(Math.round(viewportHeight * PREVIEW_MAX_RATIO), PREVIEW_MIN_HEIGHT, PREVIEW_MAX_ABS),
    [viewportHeight]
  );

  useEffect(() => {
    setIsExpanded(false);
    setPanelHeight(PREVIEW_DEFAULT_HEIGHT);
  }, [payload.datasetId]);

  useEffect(() => {
    setPanelHeight((prev) => clamp(prev, PREVIEW_MIN_HEIGHT, panelMaxHeight));
  }, [panelMaxHeight]);

  const summary = isActive
    ? '当前数据集已接入，后续分析与图表会基于 working_df。'
    : '历史卡片用于回看，不会影响当前会话状态。';

  return (
    <div className="w-full max-w-[860px] rounded-[18px] border border-slate-200 bg-white p-4 shadow-[0_10px_28px_rgba(15,23,42,0.05)] md:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] border border-slate-200 bg-slate-50">
            <FileSpreadsheet className="h-5 w-5 text-slate-700" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <div className="truncate text-base font-semibold text-slate-900">{payload.filename}</div>
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-500">
                {isActive ? '当前数据' : '历史记录'}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-xs font-medium text-slate-700">
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">{payload.rowCount} 行</span>
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">{payload.columnCount} 列</span>
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">预览 {payload.previewCount} 行</span>
            </div>
            <div className="mt-2 text-xs leading-6 text-slate-500">
              {summary} 分析基于 <span className="font-semibold text-slate-700">{payload.analysisBasis}</span>。
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            onClick={() => setIsExpanded((prev) => !prev)}
            variant="outline"
            className="h-9 rounded-[12px] border-slate-200 bg-white px-3 text-slate-700 hover:bg-slate-50"
          >
            {isExpanded ? '收起预览' : '展开预览'}
          </Button>
          {isActive ? (
            <>
              <Button
                type="button"
                onClick={onReplace}
                disabled={isUploading}
                variant="outline"
                className="h-9 rounded-[12px] border-slate-200 bg-white px-3 text-slate-700 hover:bg-slate-50"
              >
                <RefreshCcw className="mr-2 h-4 w-4" />
                {isUploading ? '上传中...' : '更换文件'}
              </Button>
              <Button
                type="button"
                onClick={() => onDelete(payload.datasetId)}
                disabled={isDeleting}
                variant="destructive"
                className="h-9 rounded-[12px] border border-rose-200 bg-rose-50 px-3 text-rose-700 hover:bg-rose-100"
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {isDeleting ? '删除中...' : '删除文件'}
              </Button>
            </>
          ) : (
            <div className="flex items-center rounded-[12px] border border-slate-200 bg-slate-50 px-3 text-[11px] font-medium text-slate-500">
              历史卡片仅用于查看，不执行删除
            </div>
          )}
        </div>
      </div>

      {isExpanded && (
        <ResizablePreviewPanel
          columns={payload.columns}
          rows={payload.preview}
          datasetId={payload.datasetId}
          height={panelHeight}
          onHeightChange={setPanelHeight}
        />
      )}

      {payload.preprocessingLog.length > 0 && (
        <details className="mt-3 rounded-[14px] border border-slate-200 bg-slate-50">
          <summary className="cursor-pointer list-none px-4 py-2.5 text-sm font-medium text-slate-800">
            预处理日志
          </summary>
          <div className="border-t border-slate-200 px-4 py-3 text-sm leading-6 text-slate-600">
            <ul className="space-y-1">
              {payload.preprocessingLog.map((item) => (
                <li key={`${payload.datasetId}-${item}`}>• {item}</li>
              ))}
            </ul>
          </div>
        </details>
      )}
    </div>
  );
}
