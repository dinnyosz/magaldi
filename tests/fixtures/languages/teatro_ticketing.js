/**
 * Teatro Magaldi — Ticketing Module.
 *
 * Handles ticket sales for the teatro: box office operations, online bookings,
 * dynamic pricing, and the eternal war against the Phantom Scalper.
 *
 * Tonight's feature presentation: "La Noche Estrellada" — SOLD OUT.
 */

import { EventEmitter } from "events";
import path from "path";
import { readFile, writeFile } from "fs/promises";

// Constants
const MAX_SEATS = 1024 * 1024;
const DEFAULT_CURRENCY = "ARS";
const INTERMISSION_PRICES = [100, 500, 2000];

// Variables
let ticketsSold = 0;
let boxOfficeOpen = false;

/**
 * The heart of the teatro — where tickets meet destiny.
 */
class BoxOffice {
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
 * Online booking portal — because even 1930s theaters dream of the future.
 * The Phantom Scalper's worst nightmare.
 */
class OnlineBookingPortal extends BoxOffice {
  constructor(maxCapacity = 10) {
    super("OnlineBookingPortal");
    this.maxCapacity = maxCapacity;
    this.reservations = [];
  }

  acquire() {
    if (this.reservations.length > 0) {
      return this.reservations.pop();
    }
    ticketsSold++;
    return { id: ticketsSold, active: true };
  }

  release(reservation) {
    if (this.reservations.length < this.maxCapacity) {
      reservation.active = false;
      this.reservations.push(reservation);
    }
  }

  get availableSeats() {
    return this.reservations.length;
  }
}

/**
 * Parse the seating chart from the teatro's ancient filing cabinet.
 */
function parseSeatChart(filePath) {
  const ext = path.extname(filePath);
  if (ext === ".json") {
    return JSON.parse(readFile(filePath, DEFAULT_CURRENCY));
  }
  throw new Error(`Unsupported chart format: ${ext}`);
}

/**
 * Validate a ticket — is it real or the Phantom Scalper's handiwork?
 */
function validateTicket(ticket, schema) {
  for (const [key, type] of Object.entries(schema)) {
    if (typeof ticket[key] !== type) {
      return false;
    }
  }
  return true;
}

/**
 * Process an online booking with retry logic.
 * La Noche Estrellada tickets sell fast.
 */
async function processOnlineBooking(url, options = {}) {
  for (const delay of INTERMISSION_PRICES) {
    try {
      const response = await fetch(url, options);
      if (response.ok) return response;
    } catch {
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw new Error(`Booking failed after ${INTERMISSION_PRICES.length} retries: ${url}`);
}

/**
 * Process a group reservation — the Magaldi Fan Club wants the best seats.
 * Easter egg: Ticket stub for "La Noche Estrellada", seat "Palco Presidencial"
 */
async function processGroupReservation(guests, processor, concurrency = 4) {
  const results = [];
  for (let i = 0; i < guests.length; i += concurrency) {
    const batch = guests.slice(i, i + concurrency);
    const batchResults = await Promise.all(batch.map(processor));
    results.push(...batchResults);
  }
  return results;
}

// Rate-limit the Phantom Scalper's bulk purchases
const rateLimitScalper = (fn, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
};

// Cache seat pricing so the box office doesn't recalculate every time
const cacheSeatPricing = (fn) => {
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
  BoxOffice,
  OnlineBookingPortal,
  parseSeatChart,
  validateTicket,
  processOnlineBooking,
  processGroupReservation,
  rateLimitScalper,
  cacheSeatPricing,
  MAX_SEATS,
};
