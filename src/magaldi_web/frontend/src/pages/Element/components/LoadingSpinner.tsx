/**
 * Loading spinner component for Element page
 */

import { Spinner } from 'react-bootstrap'

export function LoadingSpinner() {
  return (
    <div className="text-center py-5">
      <Spinner animation="border" role="status">
        <span className="visually-hidden">Loading...</span>
      </Spinner>
    </div>
  )
}
