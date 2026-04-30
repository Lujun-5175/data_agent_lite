import { afterEach, describe, expect, it, vi } from 'vitest';

describe('api config', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('supports env override for API_BASE_URL', async () => {
    vi.resetModules();
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/');

    const module = await import('./api');
    expect(module.API_BASE_URL).toBe('https://api.example.com');
    expect(module.API_ENDPOINTS.UPLOAD).toBe('https://api.example.com/upload');
  });

  it('uses localhost fallback only in dev/test mode', async () => {
    vi.resetModules();
    vi.stubEnv('VITE_API_BASE_URL', '');

    const module = await import('./api');
    expect(module.API_BASE_URL).toBe('http://127.0.0.1:8002');
  });

  it('uses same-origin proxy in production builds', async () => {
    vi.resetModules();
    vi.stubEnv('DEV', false);
    vi.stubEnv('PROD', true);
    vi.stubEnv('VITE_API_BASE_URL', 'https://dataagentlite-production.up.railway.app');

    const module = await import('./api');
    expect(module.API_BASE_URL).toBe('/api');
    expect(module.API_ENDPOINTS.UPLOAD).toBe('/api/upload');
  });

  it('maps known backend error codes', async () => {
    const { getFriendlyErrorMessage } = await import('./api');
    expect(getFriendlyErrorMessage('file_too_large', 'fallback')).toContain('50MB');
    expect(getFriendlyErrorMessage(undefined, 'fallback')).toBe('fallback');
  });
});
