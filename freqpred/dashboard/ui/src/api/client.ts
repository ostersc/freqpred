async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(path, options)
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText)
    // Try to extract the FastAPI `detail` field for a cleaner error message.
    try {
      const json = JSON.parse(text)
      if (typeof json?.detail === 'string') throw new Error(json.detail)
    } catch (e) {
      if (e instanceof SyntaxError) throw new Error(`${resp.status} ${text}`)
      throw e
    }
  }
  return resp.json() as Promise<T>
}

export default apiFetch
