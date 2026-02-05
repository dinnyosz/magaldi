/**
 * API client for Magaldi backend
 *
 * Re-exports all API functions and types for convenient importing.
 */

// Client utilities
export { API_BASE, fetchJson } from './client'

// All types
export * from './types'

// API functions by domain
export { getDashboard } from './dashboard'

export { search, generateSearchSummary } from './search'

export {
  getRepositories,
  getFileTree,
  getFileDetail,
  getVectorMap,
  getClusters,
  getFeatureGraph,
} from './repos'

export { getElement, getSimilarElements, getElementFeatures } from './elements'

export {
  getHealth,
  getJobs,
  getIndexStats,
  getMCPAnalytics,
  getMCPActivityHistory,
  clearMCPAnalytics,
  getMCPTransitionDetails,
  getMCPRecentCalls,
  getMCPCausalStatistics,
} from './admin'

export {
  getBrowseFilters,
  browseElements,
  getBrowseStats,
  getElementChildren,
  getElementDetails,
} from './browse'

export {
  getGlossaryTerms,
  getGlossaryTerm,
  searchGlossaryTerms,
  getGlossaryTermsForFeature,
} from './glossary'

export {
  getElementCallers,
  getCallChain,
  getDeadCode,
  getEntryPoints,
  getFileDependencies,
  getModuleDependents,
  getDependencyGraph,
  explainElement,
  getComplexFunctions,
  getSecurityIssues,
  getUndocumentedFunctions,
  getEnvUsage,
  getAsyncCode,
} from './analysis'
