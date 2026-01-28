import { useEffect } from 'react'

// Dark mode only - light mode has been removed
export function useDarkMode(): void {
  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', 'dark')
  }, [])
}
