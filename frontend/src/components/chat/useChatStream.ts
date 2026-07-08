import { useRef, useState } from 'react';
import { toast } from 'sonner';
import { API_ENDPOINTS, getFriendlyErrorMessage, readApiErrorInfo } from '../../config/api';
import type { UploadedDataset } from '../../types/data';
import type { ChatMessage } from './types';
import {
  extractImageUrl,
  extractStreamText,
  isNonEmptyString,
  normalizeRouteInfo,
  parseSseBlocks,
  parseSseEventBlock,
} from './utils';

export function useChatStream({
  uploadedDataset,
  buildHistory,
  appendMessage,
  upsertMessage,
  appendImageMessage,
}: {
  uploadedDataset: UploadedDataset | null;
  buildHistory: () => Array<{ type: string; content: string }>;
  appendMessage: (message: ChatMessage) => void;
  upsertMessage: (messageId: string, updater: (message: ChatMessage) => ChatMessage) => void;
  appendImageMessage: (message: ChatMessage) => void;
}) {
  const [isLoading, setIsLoading] = useState(false);
  const assistantMessageIdRef = useRef<string | null>(null);
  const statusMessageIdRef = useRef<string | null>(null);
  const activeRequestControllerRef = useRef<AbortController | null>(null);
  const requestGenerationRef = useRef(0);

  const isStaleRequest = (requestGeneration: number) => requestGeneration !== requestGenerationRef.current;

  const cancelActiveRequest = () => {
    activeRequestControllerRef.current?.abort();
    activeRequestControllerRef.current = null;
  };

  const upsertStatusMessage = (assistantMessageId: string, content: string) => {
    if (!statusMessageIdRef.current) {
      const statusMessageId = `${assistantMessageId}-status`;
      statusMessageIdRef.current = statusMessageId;
      appendMessage({
        id: statusMessageId,
        type: 'assistant',
        kind: 'status',
        content,
        timestamp: new Date(),
      });
      return;
    }

    upsertMessage(statusMessageIdRef.current, (message) => ({ ...message, content, kind: 'status' }));
  };

  const handleSendMessage = async (content: string) => {
    const outgoingContent = content.trim();
    if (!outgoingContent || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: outgoingContent,
      timestamp: new Date(),
      kind: 'text',
    };
    appendMessage(userMessage);
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();
    assistantMessageIdRef.current = assistantMessageId;
    statusMessageIdRef.current = null;
    appendMessage({
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      kind: 'text',
    });

    let accumulatedContent = '';
    const requestGeneration = requestGenerationRef.current + 1;
    requestGenerationRef.current = requestGeneration;
    cancelActiveRequest();
    const abortController = new AbortController();
    activeRequestControllerRef.current = abortController;

    try {
      const datasetId = uploadedDataset?.datasetId ?? null;
      const requestBody: Record<string, unknown> = {
        ...(datasetId ? { dataset_id: datasetId } : {}),
        input: { messages: [...buildHistory(), { type: 'human', content: outgoingContent }] },
        config: { configurable: datasetId ? { dataset_id: datasetId } : {} },
      };

      const response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });
      if (isStaleRequest(requestGeneration)) {
        return;
      }
      if (!response.ok) {
        const { code, message, requestId } = await readApiErrorInfo(response, 'Failed to call AI assistant');
        if (requestId) {
          console.error('chat request failed', { code, requestId });
        }
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
          if (isStaleRequest(requestGeneration)) {
            await reader.cancel();
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const parsedBuffer = parseSseBlocks(buffer);
          buffer = parsedBuffer.remainder;

          for (const block of parsedBuffer.blocks) {
            if (!block.trim()) continue;
            const event = parseSseEventBlock(block);
            if (!event) continue;

            if (event.eventType === 'message_chunk') {
              const chunkText = extractStreamText(event.payload);
              if (chunkText) {
                if (isStaleRequest(requestGeneration)) return;
                accumulatedContent += chunkText;
                upsertMessage(assistantMessageId, (message) => ({ ...message, content: accumulatedContent, kind: 'text' }));
              }
              continue;
            }

            if (event.eventType === 'route_info') {
              if (isStaleRequest(requestGeneration)) return;
              const routeInfo = normalizeRouteInfo(event.payload);
              if (routeInfo) {
                upsertMessage(assistantMessageId, (message) => ({ ...message, routeInfo }));
              }
              continue;
            }

            if (event.eventType === 'tool_start') {
              if (isStaleRequest(requestGeneration)) return;
              upsertStatusMessage(assistantMessageId, 'Generating analysis…');
              continue;
            }

            if (event.eventType === 'tool_end') {
              if (isStaleRequest(requestGeneration)) return;
              upsertStatusMessage(assistantMessageId, 'Analysis complete');
              continue;
            }

            if (event.eventType === 'image_generated') {
              if (isStaleRequest(requestGeneration)) return;
              const imageUrl = extractImageUrl(event.payload);
              if (imageUrl) {
                appendImageMessage({
                  id: `${assistantMessageId}-image-${Date.now()}`,
                  type: 'assistant',
                  kind: 'image',
                  content: isNonEmptyString(event.payload.filename) ? `Chart: ${event.payload.filename}` : 'Chart result',
                  imageUrl,
                  filename: isNonEmptyString(event.payload.filename) ? event.payload.filename : undefined,
                  timestamp: new Date(),
                });
              }
              upsertStatusMessage(assistantMessageId, 'Chart attached to conversation');
              continue;
            }

            if (event.eventType === 'error') {
              if (isNonEmptyString(event.payload.request_id) || isNonEmptyString(event.payload.code)) {
                console.error('chat stream error', {
                  code: isNonEmptyString(event.payload.code) ? event.payload.code : undefined,
                  requestId: isNonEmptyString(event.payload.request_id) ? event.payload.request_id : undefined,
                  stage: isNonEmptyString(event.payload.stage) ? event.payload.stage : undefined,
                  retryable: typeof event.payload.retryable === 'boolean' ? event.payload.retryable : undefined,
                });
              }
              const errorMessage = getFriendlyErrorMessage(
                isNonEmptyString(event.payload.code) ? event.payload.code : undefined,
                isNonEmptyString(event.payload.message) ? event.payload.message : 'Failed to call AI assistant'
              );
              throw new Error(errorMessage);
            }

            if (event.eventType === 'done') {
              finished = true;
              break;
            }
          }
        }
      }
    } catch (error) {
      if (abortController.signal.aborted || isStaleRequest(requestGeneration)) {
        return;
      }
      const message =
        error instanceof Error
          ? error.message === 'Failed to fetch'
              ? 'Failed to connect to the backend. Make sure your production server is running and BACKEND_URL is configured in Vercel.'
            : error.message
          : 'Error calling AI assistant'

      upsertMessage(assistantMessageId, (assistantMessage) => ({
        ...assistantMessage,
        content: `Sorry, ${message}`,
        kind: 'error',
      }));
      toast.error(message);
    } finally {
      if (!isStaleRequest(requestGeneration)) {
        activeRequestControllerRef.current = null;
        setIsLoading(false);
      }
    }
  };

  const resetChatStreamState = () => {
    requestGenerationRef.current += 1;
    cancelActiveRequest();
    setIsLoading(false);
    assistantMessageIdRef.current = null;
    statusMessageIdRef.current = null;
  };

  return {
    isLoading,
    handleSendMessage,
    resetChatStreamState,
  };
}
