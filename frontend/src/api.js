const API_BASE = '/api'

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  const config = {
    headers: {},
    ...options,
  }

  if (import.meta.env.VITE_API_KEY) {
    config.headers['X-API-Key'] = import.meta.env.VITE_API_KEY
  }

  if (config.body && !(config.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(url, config)

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP ${res.status}`)
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
