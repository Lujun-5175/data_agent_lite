const http = require('node:http');
const https = require('node:https');

const LOCALHOST_PATTERN = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/i;

function resolveBackendOrigin() {
  const candidates = [
    process.env.BACKEND_URL,
    process.env.BACKEND_ORIGIN,
    process.env.RAILWAY_BACKEND_URL,
    process.env.VITE_API_BASE_URL,
  ];

  for (const value of candidates) {
    if (!value) continue;
    const trimmed = value.trim().replace(/\/+$/, '');
    if (!trimmed || LOCALHOST_PATTERN.test(trimmed)) continue;
    return trimmed;
  }

  return null;
}

module.exports = function proxyToBackend(req, res) {
  const backendOrigin = resolveBackendOrigin();
  if (!backendOrigin) {
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(
      JSON.stringify({
        error: {
          code: 'backend_not_configured',
          message: '后端地址未配置。请在 Vercel 中设置 BACKEND_URL 为 Railway 后端地址。',
        },
      })
    );
    return;
  }

  const incomingUrl = new URL(req.url || '/', 'http://localhost');
  const proxiedPath = incomingUrl.pathname.replace(/^\/api/, '') || '/';
  const targetUrl = new URL(`${proxiedPath}${incomingUrl.search}`, backendOrigin);
  const transport = targetUrl.protocol === 'http:' ? http : https;
  const headers = { ...req.headers };

  delete headers.host;
  delete headers['x-vercel-id'];

  const proxyRequest = transport.request(
    targetUrl,
    {
      method: req.method,
      headers,
    },
    (proxyResponse) => {
      res.statusCode = proxyResponse.statusCode || 502;
      for (const [key, value] of Object.entries(proxyResponse.headers)) {
        if (value !== undefined) {
          res.setHeader(key, value);
        }
      }
      proxyResponse.pipe(res);
    }
  );

  proxyRequest.on('error', () => {
    if (!res.headersSent) {
      res.statusCode = 502;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
    }
    res.end(
      JSON.stringify({
        error: {
          code: 'backend_proxy_failed',
          message: '后端服务连接失败，请确认 Railway 后端正在运行。',
        },
      })
    );
  });

  req.pipe(proxyRequest);
};
