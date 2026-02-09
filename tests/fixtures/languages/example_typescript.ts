/**
 * Example TypeScript file exercising all extractable element types.
 *
 * This fixture ensures the magaldi index always contains TypeScript-specific
 * elements (interface, type_alias, enum, generics) when parsing its own repo.
 */

import { EventEmitter } from "events";
import type { ReadStream } from "fs";

// Constants
const API_VERSION = "v2";
const MAX_PAGE_SIZE = 100;
const ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE"] as const;

// Variables
let requestCount = 0;

// --- Type Aliases ---

type HttpMethod = (typeof ALLOWED_METHODS)[number];

type Handler<T> = (request: Request, context: Context) => Promise<T>;

type Middleware = (
  next: Handler<Response>
) => Handler<Response>;

type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// --- Enums ---

enum LogLevel {
  DEBUG = "debug",
  INFO = "info",
  WARN = "warn",
  ERROR = "error",
}

enum StatusCode {
  OK = 200,
  CREATED = 201,
  BAD_REQUEST = 400,
  NOT_FOUND = 404,
  INTERNAL_ERROR = 500,
}

const enum Direction {
  UP,
  DOWN,
  LEFT,
  RIGHT,
}

// --- Interfaces ---

interface Serializable {
  serialize(): Buffer;
  deserialize(data: Buffer): void;
}

interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(options?: QueryOptions): Promise<T[]>;
  create(entity: Omit<T, "id">): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<boolean>;
}

interface QueryOptions {
  limit?: number;
  offset?: number;
  orderBy?: string;
  direction?: "asc" | "desc";
}

interface Logger {
  level: LogLevel;
  log(level: LogLevel, message: string, meta?: Record<string, unknown>): void;
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string, error?: Error): void;
}

interface CacheEntry<T> {
  value: T;
  expiry: number;
  tags: string[];
}

// --- Classes ---

class Request {
  constructor(
    public method: HttpMethod,
    public path: string,
    public headers: Record<string, string> = {},
    public body?: unknown
  ) {}

  get isJson(): boolean {
    return this.headers["content-type"]?.includes("application/json") ?? false;
  }
}

class Response {
  constructor(
    public status: StatusCode,
    public body: unknown,
    public headers: Record<string, string> = {}
  ) {}

  static ok(body: unknown): Response {
    return new Response(StatusCode.OK, body);
  }

  static notFound(message: string = "Not found"): Response {
    return new Response(StatusCode.NOT_FOUND, { error: message });
  }
}

class Context {
  private store = new Map<string, unknown>();

  set<T>(key: string, value: T): void {
    this.store.set(key, value);
  }

  get<T>(key: string): T | undefined {
    return this.store.get(key) as T | undefined;
  }
}

class InMemoryCache<T> implements Serializable {
  private entries = new Map<string, CacheEntry<T>>();

  get(key: string): T | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiry) {
      this.entries.delete(key);
      return undefined;
    }
    return entry.value;
  }

  set(key: string, value: T, ttlMs: number, tags: string[] = []): void {
    this.entries.set(key, {
      value,
      expiry: Date.now() + ttlMs,
      tags,
    });
  }

  invalidateByTag(tag: string): number {
    let count = 0;
    for (const [key, entry] of this.entries) {
      if (entry.tags.includes(tag)) {
        this.entries.delete(key);
        count++;
      }
    }
    return count;
  }

  serialize(): Buffer {
    return Buffer.from(JSON.stringify([...this.entries]));
  }

  deserialize(data: Buffer): void {
    const parsed = JSON.parse(data.toString());
    this.entries = new Map(parsed);
  }
}

// --- Functions ---

function createRouter(): Map<string, Handler<Response>> {
  return new Map();
}

function composeMiddleware(...middlewares: Middleware[]): Middleware {
  return (next) =>
    middlewares.reduceRight((handler, mw) => mw(handler), next);
}

async function handleRequest(
  request: Request,
  handler: Handler<Response>,
  middleware: Middleware[] = []
): Promise<Response> {
  const composed = composeMiddleware(...middleware);
  const finalHandler = composed(handler);
  const context = new Context();
  return finalHandler(request, context);
}

function isSuccess(result: Result<unknown>): result is { ok: true; value: unknown } {
  return result.ok;
}

async function retry<T>(
  fn: () => Promise<T>,
  attempts: number = 3,
  delayMs: number = 1000
): Promise<Result<T>> {
  for (let i = 0; i < attempts; i++) {
    try {
      const value = await fn();
      return { ok: true, value };
    } catch (error) {
      if (i === attempts - 1) {
        return { ok: false, error: error as Error };
      }
      await new Promise((r) => setTimeout(r, delayMs * (i + 1)));
    }
  }
  return { ok: false, error: new Error("Exhausted retries") };
}

export {
  type HttpMethod,
  type Handler,
  type Result,
  LogLevel,
  StatusCode,
  Request,
  Response,
  Context,
  InMemoryCache,
  createRouter,
  handleRequest,
  retry,
};
