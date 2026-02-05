/**
 * Entry point detection patterns for various frameworks and languages
 */

export interface EntryPointConfig {
  decorators: string[]
  label: string
  icon: string
  color: string
}

export const entryPointPatterns: Record<string, EntryPointConfig> = {
  http: {
    decorators: [
      // Python: Flask, FastAPI, Django REST
      'app.route',
      'router.get',
      'router.post',
      'router.put',
      'router.delete',
      'router.patch',
      'api_view',
      'action',
      // JavaScript/TypeScript: NestJS, Express decorators
      '@get',
      '@post',
      '@put',
      '@delete',
      '@patch',
      '@controller',
      '@requestmapping',
      // PHP: Symfony attributes
      '#[route',
      '#[get',
      '#[post',
      '#[put',
      '#[delete',
      // Rust: Actix-web, Rocket, Axum
      '#[get(',
      '#[post(',
      '#[put(',
      '#[delete(',
      '#[route(',
      'actix_web::get',
      'actix_web::post',
      'rocket::get',
      'rocket::post',
      // Generic patterns
      'route',
      'endpoint',
      'api',
    ],
    label: 'HTTP Endpoint',
    icon: 'bi-globe',
    color: 'primary',
  },
  cli: {
    decorators: [
      // Python: Click, Typer
      'click.command',
      'click.group',
      'typer.command',
      // Rust: Clap
      '#[command',
      '#[clap',
      // Generic
      'command',
      'subcommand',
    ],
    label: 'CLI Command',
    icon: 'bi-terminal',
    color: 'success',
  },
  test: {
    decorators: [
      // Python: pytest
      'pytest.fixture',
      'fixture',
      // Rust
      '#[test',
      '#[tokio::test',
      '#[async_std::test',
      // PHP: PHPUnit
      '@test',
      '#[test',
      // JavaScript/TypeScript: Jest, Mocha (usually function names, but some use decorators)
      '@test',
      '@it',
      '@describe',
    ],
    label: 'Test',
    icon: 'bi-check2-circle',
    color: 'info',
  },
  async_task: {
    decorators: [
      // Python: Celery, RQ, Dramatiq
      'celery.task',
      'dramatiq.actor',
      'rq.job',
      // JavaScript: Bull, Agenda
      '@processor',
      '@process',
      '@queue',
      // Generic
      'task',
      'job',
      'worker',
      'background',
    ],
    label: 'Async Task',
    icon: 'bi-lightning',
    color: 'warning',
  },
  event: {
    decorators: [
      // JavaScript/TypeScript: EventEmitter, NestJS
      '@on',
      '@subscribe',
      '@eventhandler',
      '@listener',
      // PHP: Symfony
      '#[aseventslistener',
      // Generic
      'event',
      'handler',
      'listener',
    ],
    label: 'Event Handler',
    icon: 'bi-broadcast',
    color: 'secondary',
  },
  scheduled: {
    decorators: [
      // Python: APScheduler, Celery beat
      '@scheduled',
      'cron',
      'interval',
      // JavaScript: NestJS
      '@cron',
      '@interval',
      // Generic
      'schedule',
      'periodic',
    ],
    label: 'Scheduled',
    icon: 'bi-clock',
    color: 'dark',
  },
}

/** Main function names that indicate entry points */
export const mainFunctionNames = [
  'main',
  '__main__',
  'run',
  'start',
  'bootstrap',
  'init',
]
