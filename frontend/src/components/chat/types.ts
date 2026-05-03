import type { UploadedDataset } from '../../types/data';

export type MessageKind = 'text' | 'status' | 'error' | 'image' | 'dataset_card';

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  kind: MessageKind;
  imageUrl?: string;
  filename?: string;
  datasetPayload?: UploadedDataset;
}
