import { useRef, useState } from 'react';
import { toast } from 'sonner';
import { API_ENDPOINTS, getFriendlyErrorMessage, readApiErrorInfo } from '../../config/api';
import { SAMPLE_DATASETS, type SampleDataset } from '../../config/sampleDatasets';
import type { ServerUploadResponse, UploadedDataset } from '../../types/data';
import type { ChatMessage } from './types';
import { normalizeUploadedDataset } from './utils';

export function useDatasetUpload({
  appendMessage,
  replaceDatasetCard,
  clearDatasetCard,
  resetFileInput,
}: {
  appendMessage: (message: ChatMessage) => void;
  replaceDatasetCard: (dataset: UploadedDataset) => void;
  clearDatasetCard: () => void;
  resetFileInput: () => void;
}) {
  const [isUploading, setIsUploading] = useState(false);
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<UploadedDataset | null>(null);
  const [suggestedPrompts, setSuggestedPrompts] = useState<string[]>([]);
  const lifecycleGenerationRef = useRef(0);
  const uploadRequestIdRef = useRef(0);
  const sampleRequestIdRef = useRef(0);
  const uploadControllerRef = useRef<AbortController | null>(null);
  const sampleControllerRef = useRef<AbortController | null>(null);

  const cancelActiveTransfers = () => {
    uploadControllerRef.current?.abort();
    sampleControllerRef.current?.abort();
    uploadControllerRef.current = null;
    sampleControllerRef.current = null;
  };

  const handleFileSelect = async (file: File, options?: { sample?: SampleDataset }) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      const message = '请上传 CSV 格式的文件';
      toast.error(message);
      appendMessage({ id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() });
      return;
    }

    setIsUploading(true);
    uploadRequestIdRef.current += 1;
    const uploadRequestId = uploadRequestIdRef.current;
    const lifecycleGeneration = lifecycleGenerationRef.current;
    uploadControllerRef.current?.abort();
    const uploadController = new AbortController();
    uploadControllerRef.current = uploadController;

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(API_ENDPOINTS.UPLOAD, { method: 'POST', body: formData, signal: uploadController.signal });
      if (uploadRequestId !== uploadRequestIdRef.current || lifecycleGeneration !== lifecycleGenerationRef.current) {
        return;
      }
      if (!response.ok) {
        const { code, message } = await readApiErrorInfo(response, '文件上传失败，请检查后端服务是否正常运行');
        throw new Error(getFriendlyErrorMessage(code, message));
      }

      const result = (await response.json()) as ServerUploadResponse;
      if (uploadRequestId !== uploadRequestIdRef.current || lifecycleGeneration !== lifecycleGenerationRef.current) {
        return;
      }
      const dataset = normalizeUploadedDataset(result, file.name);
      setUploadedDataset(dataset);
      setSuggestedPrompts(options?.sample?.prompts ?? []);
      replaceDatasetCard(dataset);
      toast.success(options?.sample ? `示例数据已加载：${options.sample.name}` : result.message || `成功加载文件【${dataset.filename}】`);
      resetFileInput();
    } catch (error) {
      if (uploadController.signal.aborted || lifecycleGeneration !== lifecycleGenerationRef.current || uploadRequestId !== uploadRequestIdRef.current) {
        return;
      }
      const message = error instanceof Error ? error.message : '文件上传失败，请检查后端服务是否正常运行';
      appendMessage({ id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() });
      toast.error(message);
    } finally {
      if (uploadRequestId === uploadRequestIdRef.current && lifecycleGeneration === lifecycleGenerationRef.current) {
        uploadControllerRef.current = null;
        setIsUploading(false);
      }
    }
  };

  const handleSampleDatasetSelect = async (sampleId: string) => {
    const sample = SAMPLE_DATASETS.find((item) => item.id === sampleId);
    if (!sample || isUploading || loadingSampleId) return;

    setLoadingSampleId(sample.id);
    sampleRequestIdRef.current += 1;
    const sampleRequestId = sampleRequestIdRef.current;
    const lifecycleGeneration = lifecycleGenerationRef.current;
    sampleControllerRef.current?.abort();
    const sampleController = new AbortController();
    sampleControllerRef.current = sampleController;

    try {
      const response = await fetch(sample.path, { signal: sampleController.signal });
      if (sampleRequestId !== sampleRequestIdRef.current || lifecycleGeneration !== lifecycleGenerationRef.current) {
        return;
      }
      if (!response.ok) {
        throw new Error(`示例数据加载失败：${sample.filename}`);
      }
      const csvBlob = await response.blob();
      if (sampleRequestId !== sampleRequestIdRef.current || lifecycleGeneration !== lifecycleGenerationRef.current) {
        return;
      }
      const csvFile = new File([csvBlob], sample.filename, { type: 'text/csv' });
      await handleFileSelect(csvFile, { sample });
    } catch (error) {
      if (sampleController.signal.aborted || lifecycleGeneration !== lifecycleGenerationRef.current || sampleRequestId !== sampleRequestIdRef.current) {
        return;
      }
      const message = error instanceof Error ? error.message : '示例数据加载失败';
      appendMessage({ id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() });
      toast.error(message);
    } finally {
      if (sampleRequestId === sampleRequestIdRef.current && lifecycleGeneration === lifecycleGenerationRef.current) {
        sampleControllerRef.current = null;
        setLoadingSampleId(null);
      }
    }
  };

  const handleDeleteDataset = async (targetDatasetId: string) => {
    if (!targetDatasetId) return;
    lifecycleGenerationRef.current += 1;
    cancelActiveTransfers();
    setIsUploading(false);
    setLoadingSampleId(null);
    setIsDeleting(true);
    try {
      const response = await fetch(API_ENDPOINTS.DELETE_DATASET(targetDatasetId), { method: 'DELETE' });
      if (!response.ok) {
        const { code, message } = await readApiErrorInfo(response, '删除数据集失败，请稍后重试');
        throw new Error(getFriendlyErrorMessage(code, message));
      }
      setUploadedDataset(null);
      setSuggestedPrompts([]);
      clearDatasetCard();
      appendMessage({
        id: `status-${Date.now()}`,
        type: 'assistant',
        kind: 'status',
        content: '当前数据集已移除，你可以继续聊天或重新上传文件。',
        timestamp: new Date(),
      });
      toast.success('数据集已删除');
    } catch (error) {
      const message = error instanceof Error ? error.message : '删除数据集失败，请稍后重试';
      appendMessage({ id: `error-${Date.now()}`, type: 'assistant', kind: 'error', content: message, timestamp: new Date() });
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  const resetDatasetState = () => {
    lifecycleGenerationRef.current += 1;
    cancelActiveTransfers();
    setIsUploading(false);
    setLoadingSampleId(null);
    setIsDeleting(false);
    setUploadedDataset(null);
    setSuggestedPrompts([]);
  };

  return {
    isUploading,
    loadingSampleId,
    isDeleting,
    uploadedDataset,
    suggestedPrompts,
    handleFileSelect,
    handleSampleDatasetSelect,
    handleDeleteDataset,
    resetDatasetState,
  };
}
