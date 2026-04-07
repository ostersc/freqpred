async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(path, options)
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText)
    throw new Error(`${resp.status} ${text}`)
  }
  return resp.json() as Promise<T>
}

export default apiFetch
