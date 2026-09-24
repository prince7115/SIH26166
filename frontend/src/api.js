import { state } from './state.js';

// ngrok's free tier serves a warning page instead of the API unless this header is sent.
const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

function apiUrl(endpoint) {
  return state.colabUrl.replace(/\/+$/, '') + endpoint;
}

export async function apiFetch(endpoint) {
  const resp = await fetch(apiUrl(endpoint), { headers: NGROK_HEADERS });
  if (!resp.ok) throw new Error(`API error: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

export async function ping(timeoutMs) {
  const resp = await fetch(apiUrl('/api/connect'), {
    headers: NGROK_HEADERS,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!resp.ok) throw new Error('Server returned error');
}

// Server-sent events read with fetch, because EventSource cannot send the ngrok header.
export async function streamEvents(endpoint, onEvent) {
  const resp = await fetch(apiUrl(endpoint), { headers: NGROK_HEADERS });
  if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();   // an incomplete line waits for the next chunk
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch (e) {
        console.warn('SSE parse error:', e, line);
      }
    }
  }
}

export async function fetchBlob(endpoint) {
  const resp = await fetch(apiUrl(endpoint), { headers: NGROK_HEADERS });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `Download failed (${resp.status})`);
  }
  return resp.blob();
}
