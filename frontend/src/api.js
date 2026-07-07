const API_BASE = '/api'

let _apiKey = null

async function initApiKey() {
  try {
    const res = await fetch(`${API_BASE}/config`)
    const data = await res.json()
    if (data.has_api_key && data.api_key) {
      _apiKey = data.api_key
    }
  } catch {
    // backend will handle missing key gracefully if not configured
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  const config = {
    headers: {},
    ...options,
  }

  if (_apiKey) {
    config.headers['X-API-Key'] = _apiKey
  }

  if (config.body && !(config.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(url, config)

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const error = await res.json()
      if (error.detail) detail = error.detail
    } catch { /* ignore */ }
    throw new Error(detail)
  }

  return res.json()
}

export async function queryDocuments(question) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({ question }),
  })
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request('/ingest', {
    method: 'POST',
    body: formData,
  })
}

export async function getStats() {
  return request('/stats')
}

export async function getDocuments() {
  return request('/documents')
}

export async function healthCheck() {
  return request('/health')
}

export async function getFeatures() {
  return request('/features')
}

export { initApiKey }
