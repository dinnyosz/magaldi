/**
 * Breadcrumb navigation for Element page
 */

import { Link } from 'react-router-dom'
import { Breadcrumb } from 'react-bootstrap'
import type { ElementDetail } from '../../../api'

interface Props {
  element: ElementDetail
}

export function BreadcrumbNav({ element }: Props) {
  return (
    <Breadcrumb className="mb-4">
      <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
        Repositories
      </Breadcrumb.Item>
      <Breadcrumb.Item
        linkAs={Link}
        linkProps={{
          to: `/repos/${element.repository.scope}/${element.repository.name}`,
        }}
      >
        {element.repository.scope}/{element.repository.name}
      </Breadcrumb.Item>
      {element.context.file && element.element_type !== 'file' && (
        <Breadcrumb.Item
          linkAs={Link}
          linkProps={{
            to: `/element/${encodeURIComponent(element.context.file.hash_id || element.context.file.element_id)}`,
          }}
        >
          {element.context.file.name}
        </Breadcrumb.Item>
      )}
      {element.context.parent &&
        element.context.parent.element_type !== 'file' && (
          <Breadcrumb.Item
            linkAs={Link}
            linkProps={{
              to: `/element/${encodeURIComponent(element.context.parent.hash_id || element.context.parent.element_id)}`,
            }}
          >
            {element.context.parent.name}
          </Breadcrumb.Item>
        )}
      <Breadcrumb.Item active>{element.name}</Breadcrumb.Item>
    </Breadcrumb>
  )
}
