/**
 * Custom hook for fetching element data and related information
 */

import { useQuery } from '@tanstack/react-query'
import {
  getElement,
  getSimilarElements,
  explainElement,
  getGlossaryTermsForFeature,
  getElementFeatures,
} from '../../../api'

export function useElementData(decodedId: string) {
  const {
    data: element,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['element', decodedId],
    queryFn: () => getElement(decodedId),
    enabled: !!decodedId,
  })

  const { data: similar } = useQuery({
    queryKey: ['similar', decodedId],
    queryFn: () => getSimilarElements(decodedId, 10),
    enabled: !!decodedId,
  })

  // Fetch call analysis data for functions/methods
  const { data: explanation } = useQuery({
    queryKey: ['explain', decodedId],
    queryFn: () => explainElement(decodedId),
    enabled:
      !!decodedId &&
      !!element &&
      ['function', 'method'].includes(element.element_type),
  })

  // Fetch glossary terms for features/subfeatures
  const { data: glossaryTerms } = useQuery({
    queryKey: ['glossaryForFeature', decodedId],
    queryFn: () => getGlossaryTermsForFeature(decodedId),
    enabled:
      !!decodedId &&
      !!element &&
      ['feature', 'subfeature'].includes(element.element_type),
  })

  // Fetch connected features for code elements (not features/subfeatures/glossary themselves)
  const { data: elementFeatures } = useQuery({
    queryKey: ['elementFeatures', decodedId],
    queryFn: () => getElementFeatures(decodedId),
    enabled:
      !!decodedId &&
      !!element &&
      !['feature', 'subfeature', 'glossary'].includes(element.element_type),
  })

  return {
    element,
    isLoading,
    error,
    similar,
    explanation,
    glossaryTerms,
    elementFeatures,
  }
}
