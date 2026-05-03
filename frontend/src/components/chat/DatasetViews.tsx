import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { FileSpreadsheet, GripVertical, RefreshCcw, Trash2 } from 'lucide-react';
import type { UploadedDataset } from '../../types/data';
import { Button } from '../ui/button';
import { clamp, formatCellValue, getViewportHeight } from './utils';

const PREVIEW_MIN_HEIGHT = 240;
const PREVIEW_DEFAULT_HEIGHT = 336;
const PREVIEW_MAX_RATIO = 0.58;
const PREVIEW_MAX_ABS = 620;

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

export function DataPreviewCard({
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

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
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
