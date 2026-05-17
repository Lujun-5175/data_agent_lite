import type { UploadedDataset } from '../../types/data';

export type MessageKind = 'text' | 'status' | 'error' | 'image' | 'dataset_card';

export interface RouteInfo {
  intentType: string;
  confidence: 'low' | 'medium' | 'high' | string;
  routeSource: string;
  conflictFlags: string[];
  suggestedPlan: string[];
  requiresMl: boolean;
  requiresChart: boolean;
  requiresPythonAnalysis: boolean;
  isFollowUp: boolean;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  kind: MessageKind;
  imageUrl?: string;
  filename?: string;
  datasetPayload?: UploadedDataset;
  routeInfo?: RouteInfo;
}
