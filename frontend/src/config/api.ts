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
  message?: string;
  detail?: string;
};

export async function readApiErrorInfo(response: Response, fallbackMessage: string) {
  try {
    const payload = (await response.json()) as ApiErrorPayload;
    const code = payload?.error?.code;
    const message = payload?.error?.message || payload?.message || payload?.detail || fallbackMessage;
    return { code, message };
  } catch {
    return { code: undefined, message: fallbackMessage };
  }
}

export function getFriendlyErrorMessage(code: string | undefined, fallbackMessage: string) {
  switch (code) {
    case 'dataset_required':
      return '当前未选择数据集，请先上传 CSV 文件后再进行数据分析。';
    case 'dataset_not_found':
      return '当前数据集已被删除或不存在，请重新上传数据文件。';
    case 'invalid_file_type':
      return '只支持 CSV 文件上传。';
    case 'file_too_large':
      return '上传文件超过 50MB 限制。';
    case 'invalid_python_code':
      return '分析代码被安全策略拦截，请调整后重试。';
    case 'structured_failure':
      return '本次请求没有产出可复核的结构化结果，请检查字段、目标列或图表描述后重试。';
    case 'correlation_unsupported':
      return '当前版本暂不支持该类型相关性分析。';
    case 'internal_error':
      return '服务器内部错误，请稍后重试。';
    default:
      return fallbackMessage;
  }
}
