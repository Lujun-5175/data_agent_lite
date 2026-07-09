import type { UploadedDataset } from '../../types/data';

export type MessageKind = 'text' | 'status' | 'error' | 'image' | 'dataset_card';

export interface RouteTaskInfo {
  taskId: string;
  taskType: string;
  description: string;
  dependsOn: string[];
  requiredOutputs: string[];
}

export interface RouteInfo {
  primaryMode: string;
  confidenceScore: number | null;
  intentType: string;
  confidence: 'low' | 'medium' | 'high' | string;
  routeSource: string;
  conflictFlags: string[];
  ambiguityFlags: string[];
  guardrailActions: string[];
  fallbackReasons: string[];
  suggestedPlan: string[];
  requestedCapabilities: string[];
  requiresMl: boolean;
  requiresChart: boolean;
  requiresPythonAnalysis: boolean;
  isFollowUp: boolean;
  needsDataset: boolean;
  needsToolExecution: boolean;
  needsArtifactContext: boolean;
  finalBranch: string;
  taskPlanAvailable: boolean;
  taskPlanGoal: string;
  taskPlanConfidence: number | null;
  taskPlanTasks: RouteTaskInfo[];
  taskPlanAmbiguityFlags: string[];
  taskPlanAssumptions: string[];
  taskPlanAttempted: boolean;
  taskPlanGenerationFailed: boolean;
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
  /** DeepSeek reasoning/thinking chain content */
  thinkingContent?: string;
}
