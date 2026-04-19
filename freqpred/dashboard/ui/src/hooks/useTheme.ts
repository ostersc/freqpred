import { useEffect, useState } from 'react'

function getStored(key: string, fallback: boolean): boolean {
  const v = localStorage.getItem(key)
  return v === null ? fallback : v === 'true'
}

export function useTheme() {
  const [light, setLight] = useState(() => getStored('theme-light', false))
  const [dense, setDense] = useState(() => getStored('theme-dense', false))

  useEffect(() => {
    document.body.classList.toggle('light', light)
    localStorage.setItem('theme-light', String(light))
  }, [light])

  useEffect(() => {
    document.body.classList.toggle('dense', dense)
    localStorage.setItem('theme-dense', String(dense))
  }, [dense])

  return { light, dense, toggleLight: () => setLight((v) => !v), toggleDense: () => setDense((v) => !v) }
}
