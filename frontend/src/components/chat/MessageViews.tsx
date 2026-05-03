import { Bot, FileSpreadsheet, Image as ImageIcon, Table2, TriangleAlert, UserRound } from 'lucide-react';
import type { UploadedDataset } from '../../types/data';
import type { ChatMessage } from './types';
import { DataPreviewCard } from './DatasetViews';

export function SuggestedPrompts({
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

export function SampleDataPanel({
  samples,
  loadingSampleId,
  disabled,
  onSelect,
}: {
  samples: Array<{ id: string; name: string; description: string; columnCount: number; rowCount: number }>;
  loadingSampleId: string | null;
  disabled: boolean;
  onSelect: (sampleId: string) => void;
}) {
  return (
    <section className="rounded-[24px] border border-slate-200/80 bg-white/82 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.045)] backdrop-blur-sm md:p-5">
      <div>
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
          <FileSpreadsheet className="h-3.5 w-3.5" />
          示例数据一键试用
        </div>
        <h2 className="mt-3 text-lg font-semibold tracking-tight text-slate-950 md:text-xl">选择一个示例数据集，立即开始体验</h2>
        <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600">
          不需要准备 CSV，点击加载后即可测试数据预览、统计分析、图表生成和建模流程。
        </p>
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
                onClick={() => onSelect(sample.id)}
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

export function ChatBubble({ message }: { message: ChatMessage }) {
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
        <div className="whitespace-pre-wrap break-words text-[15px] font-medium leading-7 text-slate-900">{message.content}</div>
      </div>

      {isUser && (
        <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-slate-50 shadow-[0_4px_12px_rgba(15,23,42,0.04)]">
          <UserRound className="h-4 w-4 text-slate-700" />
        </div>
      )}
    </div>
  );
}

export function ImageCard({ message }: { message: ChatMessage }) {
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

export function DatasetCardMessage({
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
  return (
    <div className="flex w-full items-start gap-3">
      <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-white shadow-[0_4px_12px_rgba(15,23,42,0.035)]">
        <Table2 className="h-4 w-4 text-slate-700" />
      </div>
      <DataPreviewCard
        payload={payload}
        isActive={isActive}
        onReplace={onReplace}
        onDelete={onDelete}
        isUploading={isUploading}
        isDeleting={isDeleting}
      />
    </div>
  );
}
