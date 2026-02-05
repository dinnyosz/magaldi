/**
 * Element type configuration for badges and icons
 */

export const typeConfig: Record<string, { icon: string; color: string; label: string }> = {
  file: { icon: 'bi-file-code', color: 'info', label: 'File' },
  class: { icon: 'bi-box', color: 'purple', label: 'Class' },
  interface: { icon: 'bi-layers', color: 'info', label: 'Interface' },
  type_alias: { icon: 'bi-type', color: 'info', label: 'Type Alias' },
  trait: { icon: 'bi-diagram-3', color: 'info', label: 'Trait' },
  enum: { icon: 'bi-list-ol', color: 'warning', label: 'Enum' },
  function: { icon: 'bi-braces', color: 'primary', label: 'Function' },
  method: { icon: 'bi-gear', color: 'success', label: 'Method' },
  variable: { icon: 'bi-x-diamond', color: 'secondary', label: 'Variable' },
  constant: { icon: 'bi-hash', color: 'warning', label: 'Constant' },
  import: { icon: 'bi-box-arrow-in-right', color: 'secondary', label: 'Import' },
  feature: { icon: 'bi-collection', color: 'info', label: 'Feature' },
  subfeature: { icon: 'bi-collection-fill', color: 'info', label: 'Subfeature' },
  glossary: { icon: 'bi-book', color: 'primary', label: 'Glossary Term' },
}

export function getTypeConfig(type: string) {
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary', label: type }
}

export function getTypeBadgeStyle(type: string): React.CSSProperties {
  if (type === 'class') {
    return { backgroundColor: '#6f42c1', color: 'white' }
  }
  return {}
}

export const languageIcons: Record<string, string> = {
  python: 'bi-filetype-py',
  javascript: 'bi-filetype-js',
  typescript: 'bi-filetype-tsx',
  php: 'bi-filetype-php',
  rust: 'bi-filetype-rs',
}
