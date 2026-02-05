/**
 * Helper functions for element page
 */

import type { ElementDetail, DecoratorDetailInfo } from '../../../api'
import { entryPointPatterns, mainFunctionNames } from './entryPoints'

export interface DetectedEntryPoint {
  label: string
  icon: string
  color: string
  args?: string
}

/**
 * Extract clean value from decorator args like ("serve") -> serve
 */
export function extractCleanArg(args: string): string {
  // Strip whitespace and outer parentheses
  let cleaned = args.trim()
  if (cleaned.startsWith('(') && cleaned.endsWith(')')) {
    cleaned = cleaned.slice(1, -1).trim()
  }
  // Extract first quoted string
  const match = cleaned.match(/^["']([^"']+)["']/)
  if (match) {
    return match[1]
  }
  return cleaned
}

/**
 * Detect entry point type from element decorators
 */
export function detectEntryPointType(
  element: ElementDetail
): DetectedEntryPoint | null {
  let entryPointType: DetectedEntryPoint | null = null

  if (element.decorators && element.decorators.length > 0) {
    for (const [, epConfig] of Object.entries(entryPointPatterns)) {
      for (const decorator of element.decorators) {
        const decoratorLower = decorator.toLowerCase()
        if (
          epConfig.decorators.some((d) =>
            decoratorLower.includes(d.toLowerCase())
          )
        ) {
          // Try to find matching decorator_details to get args
          let args: string | undefined
          if (element.decorator_details) {
            const matchingDetail = element.decorator_details.find(
              (dd: DecoratorDetailInfo) =>
                dd.name && decorator.toLowerCase().includes(dd.name.toLowerCase())
            )
            if (matchingDetail?.args) {
              args = extractCleanArg(matchingDetail.args)
            }
          }
          entryPointType = {
            label: epConfig.label,
            icon: epConfig.icon,
            color: epConfig.color,
            args,
          }
          break
        }
      }
      if (entryPointType) break
    }
  }

  // Also check for main function (common across languages)
  if (!entryPointType && element.element_type === 'function') {
    if (mainFunctionNames.includes(element.name.toLowerCase())) {
      entryPointType = {
        label: 'Main Entry',
        icon: 'bi-play-circle',
        color: 'danger',
      }
    }
  }

  return entryPointType
}

/**
 * Check if element is a callable (function or method)
 */
export function isCallable(elementType: string): boolean {
  return ['function', 'method'].includes(elementType)
}

/**
 * Check if element is a container (file or class)
 */
export function isContainer(elementType: string): boolean {
  return ['file', 'class'].includes(elementType)
}

/**
 * Check if element is a feature type
 */
export function isFeature(elementType: string): boolean {
  return ['feature', 'subfeature'].includes(elementType)
}

/**
 * Check if element is a glossary term
 */
export function isGlossary(elementType: string): boolean {
  return elementType === 'glossary'
}
