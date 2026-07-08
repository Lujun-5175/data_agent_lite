const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8002';
const PRODUCTION_API_PROXY_BASE_URL = '/api';

function resolveApiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (import.meta.env.PROD) return PRODUCTION_API_PROXY_BASE_URL;
  if (configured) {
    return configured.replace(/\/+$/, '');
  }
  return DEFAULT_API_BASE_URL;
}

export const API_BASE_URL = resolveApiBaseUrl();

const buildApiUrl = (path: string) => `${API_BASE_URL}${path}`;

export const API_ENDPOINTS = {
  UPLOAD: buildApiUrl('/upload'),
  DELETE_DATASET: (datasetId: string) => buildApiUrl(`/datasets/${datasetId}`),
  CALCULATE_CORRELATION: buildApiUrl('/calculate-correlation'),
  CHAT_STREAM: buildApiUrl('/chat/stream'),
} as const;

export const getImageUrl = (filename: string) => buildApiUrl(`/static/images/${filename}`);

export type ApiErrorPayload = {
  error?: {
    code?: string;
    message?: string;
  };
  request_id?: string;
  message?: string;
  detail?: string;
};

export async function readApiErrorInfo(response: Response, fallbackMessage: string) {
  try {
    const payload = (await response.json()) as ApiErrorPayload;
    const code = payload?.error?.code;
    const message = payload?.error?.message || payload?.message || payload?.detail || fallbackMessage;
    const requestId = payload?.request_id ?? response.headers.get('X-Request-ID') ?? undefined;
    return { code, message, requestId };
  } catch {
    return { code: undefined, message: fallbackMessage, requestId: response.headers.get('X-Request-ID') ?? undefined };
  }
}

export function getFriendlyErrorMessage(code: string | undefined, fallbackMessage: string) {
  switch (code) {
    case 'dataset_required':
      return 'No dataset selected. Please upload a CSV file first.';
    case 'dataset_not_found':
      return 'The dataset has been deleted or does not exist. Please re-upload your data.';
    case 'invalid_file_type':
      return 'Only CSV files are supported.';
    case 'file_too_large':
      return 'Uploaded file exceeds 50MB limit.';
    case 'invalid_python_code':
      return 'Analysis code was blocked by security policy. Please adjust and try again.';
    case 'structured_failure':
      return 'The request did not produce a reviewable structured result. Please check fields, target column, or chart description and try again.';
    case 'agent_recursion_limit':
      return 'The task was too complex or the execution path looped repeatedly. Please break your question down further.';
    case 'upstream_model_stream_error':
      return 'The upstream model stream was interrupted. The system attempted to recover. Please try again.';
    case 'upstream_model_timeout':
      return 'The upstream model timed out. Please try again.';
    case 'upstream_model_connection_error':
      return fallbackMessage;
    case 'upstream_model_http_error':
      return fallbackMessage;
    case 'upstream_model_request_error':
      return fallbackMessage;
    case 'tool_execution_timeout':
      return 'Analysis execution timed out. Try narrowing down columns or running a smaller analysis.';
    case 'history_compression_error':
      return 'Failed to compress conversation context. Please send a new request.';
    case 'stream_interrupted':
      return 'The response stream was interrupted. Please try again.';
    case 'correlation_unsupported':
      return 'This type of correlation analysis is not supported in the current version.';
    case 'internal_error':
      return fallbackMessage;
    default:
      return fallbackMessage;
  }
}
