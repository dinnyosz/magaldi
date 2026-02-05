/**
 * Error alert component for Element page
 */

import { Alert } from 'react-bootstrap'

interface Props {
  error: Error | null
}

export function ErrorAlert({ error }: Props) {
  return (
    <Alert variant="danger">
      Failed to load element: {error?.message || 'Unknown error'}
    </Alert>
  )
}
