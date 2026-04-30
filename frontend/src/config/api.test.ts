import { describe, expect, it, vi } from 'vitest';

describe('api config', () => {
  it('supports env override for API_BASE_URL', async () => {
    vi.resetModules();
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com');

    const module = await import('./api');
    expect(module.API_BASE_URL).toBe('https://api.example.com');
    expect(module.API_ENDPOINTS.UPLOAD).toBe('https://api.example.com/upload');
  });

  it('maps known backend error codes', async () => {
    const { getFriendlyErrorMessage } = await import('./api');
    expect(getFriendlyErrorMessage('file_too_large', 'fallback')).toContain('50MB');
    expect(getFriendlyErrorMessage(undefined, 'fallback')).toBe('fallback');
  });
});
