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
      const message = 'Please upload a CSV file.';
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
        const { code, message } = await readApiErrorInfo(response, 'File upload failed. Is the backend server running?');
        throw new Error(getFriendlyErrorMessage(code, message));
      }

      const result = (await response.json()) as ServerUploadResponse;
      if (uploadRequestId !== uploadRequestIdRef.current || lifecycleGeneration !== lifecycleGenerationRef.current) {
        return;
      }
      const dataset = normalizeUploadedDataset(result, file.name);
      setUploadedDataset(dataset);
      setSuggestedPrompts(dataset.recommendedPrompts);
      replaceDatasetCard(dataset);
      toast.success(options?.sample ? `Sample loaded: ${options.sample.name}` : result.message || `Loaded: ${dataset.filename}`);
      resetFileInput();
    } catch (error) {
      if (uploadController.signal.aborted || lifecycleGeneration !== lifecycleGenerationRef.current || uploadRequestId !== uploadRequestIdRef.current) {
        return;
      }
      const message = error instanceof Error ? error.message : 'File upload failed. Is the backend server running?';
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
        throw new Error(`Failed to load sample: ${sample.filename}`);
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
      const message = error instanceof Error ? error.message : 'Failed to load sample data';
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
        const { code, message } = await readApiErrorInfo(response, 'Failed to delete dataset. Please try again.');
        throw new Error(getFriendlyErrorMessage(code, message));
      }
      setUploadedDataset(null);
      setSuggestedPrompts([]);
      clearDatasetCard();
      appendMessage({
        id: `status-${Date.now()}`,
        type: 'assistant',
        kind: 'status',
        content: 'Dataset removed. You can continue chatting or upload a new file.',
        timestamp: new Date(),
      });
      toast.success('Dataset deleted');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete dataset. Please try again.';
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
