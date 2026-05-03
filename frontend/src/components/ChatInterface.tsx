import { useEffect, useRef, useState } from 'react';
import { Bot, Paperclip, Send } from 'lucide-react';
import { CHAT_SHELL_CLASS } from '../config/layout';
import { SAMPLE_DATASETS } from '../config/sampleDatasets';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { DatasetCardMessage } from './chat/MessageViews';
import { ChatBubble, ImageCard, SampleDataPanel, SuggestedPrompts } from './chat/MessageViews';
import type { ChatMessage } from './chat/types';
import { buildMessageHistory } from './chat/utils';
import { useChatStream } from './chat/useChatStream';
import { useDatasetUpload } from './chat/useDatasetUpload';

const initialMessages: ChatMessage[] = [
  {
    id: '1',
    type: 'assistant',
    kind: 'text',
    content: '你好，欢迎来到 Data Agent。你可以直接聊天，也可以在输入框左侧上传 CSV，让我帮你分析、预览和可视化。',
    timestamp: new Date(),
  },
];

interface ChatInterfaceProps {
  clearTrigger: number;
}

export function ChatInterface({ clearTrigger }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const appendMessage = (message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  };

  const upsertMessage = (messageId: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((message) => (message.id === messageId ? updater(message) : message)));
  };

  const replaceDatasetCard = (datasetPayload: NonNullable<ChatMessage['datasetPayload']>) => {
    setMessages((prev) => [
      ...prev.filter((message) => message.kind !== 'dataset_card'),
      {
        id: `${datasetPayload.datasetId}-dataset-card`,
        type: 'assistant',
        kind: 'dataset_card',
        content: '',
        datasetPayload,
        timestamp: new Date(),
      },
    ]);
  };

  const clearDatasetCard = () => {
    setMessages((prev) => prev.filter((message) => message.kind !== 'dataset_card'));
  };

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const {
    isUploading,
    loadingSampleId,
    isDeleting,
    uploadedDataset,
    suggestedPrompts,
    handleFileSelect,
    handleSampleDatasetSelect,
    handleDeleteDataset,
    resetDatasetState,
  } = useDatasetUpload({
    appendMessage,
    replaceDatasetCard,
    clearDatasetCard,
    resetFileInput,
  });

  const { isLoading, handleSendMessage, resetChatStreamState } = useChatStream({
    uploadedDataset,
    buildHistory: () => buildMessageHistory(messages),
    appendMessage,
    upsertMessage,
    appendImageMessage: appendMessage,
  });

  useEffect(() => {
    if (clearTrigger <= 0) return;
    setMessages(initialMessages);
    setInputValue('');
    resetDatasetState();
    resetChatStreamState();
    resetFileInput();
  }, [clearTrigger]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isLoading, uploadedDataset]);

  const handleFileUploadClick = () => fileInputRef.current?.click();

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const handleSend = async (messageOverride?: string) => {
    const content = (messageOverride ?? inputValue).trim();
    if (!content) return;
    setInputValue('');
    await handleSendMessage(content);
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      void handleFileSelect(file);
    }
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
                  onSelect={(sampleId) => void handleSampleDatasetSelect(sampleId)}
                />
              )}

              <SuggestedPrompts prompts={suggestedPrompts} disabled={isLoading} onSelect={(prompt) => void handleSend(prompt)} />

              {messages.map((message) => {
                if (message.kind === 'image') {
                  return <ImageCard key={message.id} message={message} />;
                }

                if (message.kind === 'dataset_card' && message.datasetPayload) {
                  return (
                    <DatasetCardMessage
                      key={message.id}
                      payload={message.datasetPayload}
                      isActive={uploadedDataset?.datasetId === message.datasetPayload.datasetId}
                      onReplace={handleFileUploadClick}
                      onDelete={handleDeleteDataset}
                      isUploading={isUploading}
                      isDeleting={isDeleting}
                    />
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
                  onChange={(event) => setInputValue(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入你的问题，或者先上传 CSV 文件..."
                  className="min-h-[52px] max-h-[160px] flex-1 resize-none rounded-[16px] border-0 bg-transparent px-3 py-3 text-[15px] font-medium text-slate-900 shadow-none placeholder:text-slate-400 focus-visible:ring-0 focus-visible:ring-offset-0"
                  disabled={isLoading}
                />
              </div>

              <Button
                type="button"
                onClick={() => void handleSend()}
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
