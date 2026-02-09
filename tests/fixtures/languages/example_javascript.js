/**
 * Example JavaScript file exercising all extractable element types.
 *
 * This fixture ensures the magaldi index always contains JavaScript-specific
 * elements (class, arrow functions, React patterns) when parsing its own repo.
 */

import { EventEmitter } from "events";
import path from "path";
import { readFile, writeFile } from "fs/promises";

// Constants
const MAX_BUFFER_SIZE = 1024 * 1024;
const DEFAULT_ENCODING = "utf-8";
const RETRY_DELAYS = [100, 500, 2000];

// Variables
let connectionCount = 0;
let isInitialized = false;

/**
 * Base event handler class with lifecycle hooks.
 */
class EventHandler {
  constructor(name) {
    this.name = name;
    this.listeners = new Map();
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  emit(event, data) {
    const handlers = this.listeners.get(event) || [];
    handlers.forEach((handler) => handler(data));
  }

  removeAll() {
    this.listeners.clear();
  }
}

/**
 * Connection pool with automatic retry logic.
 */
class ConnectionPool extends EventHandler {
  constructor(maxSize = 10) {
    super("ConnectionPool");
    this.maxSize = maxSize;
    this.pool = [];
  }

  acquire() {
    if (this.pool.length > 0) {
      return this.pool.pop();
    }
    connectionCount++;
    return { id: connectionCount, active: true };
  }

  release(connection) {
    if (this.pool.length < this.maxSize) {
      connection.active = false;
      this.pool.push(connection);
    }
  }

  get size() {
    return this.pool.length;
  }
}

/**
 * Parse a configuration file and return structured data.
 */
function parseConfig(filePath) {
  const ext = path.extname(filePath);
  if (ext === ".json") {
    return JSON.parse(readFile(filePath, DEFAULT_ENCODING));
  }
  throw new Error(`Unsupported format: ${ext}`);
}

/**
 * Validate input data against a schema.
 */
function validateInput(data, schema) {
  for (const [key, type] of Object.entries(schema)) {
    if (typeof data[key] !== type) {
      return false;
    }
  }
  return true;
}

/**
 * Async function to fetch data with retry.
 */
async function fetchWithRetry(url, options = {}) {
  for (const delay of RETRY_DELAYS) {
    try {
      const response = await fetch(url, options);
      if (response.ok) return response;
    } catch {
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw new Error(`Failed after ${RETRY_DELAYS.length} retries: ${url}`);
}

/**
 * Process items in parallel with concurrency limit.
 */
async function processParallel(items, processor, concurrency = 4) {
  const results = [];
  for (let i = 0; i < items.length; i += concurrency) {
    const batch = items.slice(i, i + concurrency);
    const batchResults = await Promise.all(batch.map(processor));
    results.push(...batchResults);
  }
  return results;
}

// Arrow function assigned to variable
const debounce = (fn, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
};

// Arrow function with complex logic
const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};

export {
  EventHandler,
  ConnectionPool,
  parseConfig,
  validateInput,
  fetchWithRetry,
  processParallel,
  debounce,
  memoize,
  MAX_BUFFER_SIZE,
};
