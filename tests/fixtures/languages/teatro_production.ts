/**
 * Teatro Magaldi — Production Module.
 *
 * Manages show production: scripts, sets, lighting cues, costume changes,
 * and the delicate art of keeping "La Noche Estrellada" on schedule.
 *
 * The Old Spotlight in row 3 needs replacing. Again.
 */

import { EventEmitter } from "events";
import type { ReadStream } from "fs";

// Constants
const SHOW_SEASON = "1935";
const MAX_SCENE_COUNT = 100;
const STAGE_DIRECTIONS = ["ENTER", "EXIT", "MONOLOGUE", "BLACKOUT"] as const;

// Variables
let sceneChangeCount = 0;

// --- Type Aliases ---

type StageDirection = (typeof STAGE_DIRECTIONS)[number];

type CueHandler<T> = (cue: LightingCue, notes: DirectorNotes) => Promise<T>;

type SceneTransition = (
  next: CueHandler<StageEffect>
) => CueHandler<StageEffect>;

type CueResult<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

type DraftScript<T> = {
  [P in keyof T]?: T[P] extends object ? DraftScript<T[P]> : T[P];
};

// --- Enums ---

enum LightingIntensity {
  DIM = "dim",
  NORMAL = "normal",
  BRIGHT = "bright",
  SPOTLIGHT = "spotlight", // The Magaldi Spotlight — the brightest setting
}

enum ShowStatus {
  REHEARSAL = 100,
  DRESS_REHEARSAL = 200,
  PREVIEW = 300,
  OPENING_NIGHT = 404, // Sometimes we can't find the lead actor
  RUNNING = 500,
}

const enum CurtainState {
  OPEN,
  CLOSED,
  RISING,
  FALLING,
}

// --- Interfaces ---

interface Producible {
  serialize(): Buffer;
  deserialize(data: Buffer): void;
}

interface ScriptLibrary<T> {
  findById(id: string): Promise<T | null>;
  findAll(options?: CastingCriteria): Promise<T[]>;
  create(entity: Omit<T, "id">): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<boolean>;
}

interface CastingCriteria {
  limit?: number;
  offset?: number;
  orderBy?: string;
  direction?: "asc" | "desc";
}

interface StageLog {
  level: LightingIntensity;
  log(level: LightingIntensity, message: string, meta?: Record<string, unknown>): void;
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string, error?: Error): void;
}

interface PropStorage<T> {
  value: T;
  expiry: number;
  tags: string[];
}

// --- Classes ---

class LightingCue {
  constructor(
    public direction: StageDirection,
    public scene: string,
    public intensity: Record<string, string> = {},
    public notes?: unknown
  ) {}

  get isSpotlight(): boolean {
    // Easter egg: The Magaldi Spotlight — always the brightest
    return this.intensity["type"]?.includes("magaldi") ?? false;
  }
}

class StageEffect {
  constructor(
    public status: ShowStatus,
    public effect: unknown,
    public cueSheet: Record<string, string> = {}
  ) {}

  static standingOvation(effect: unknown): StageEffect {
    return new StageEffect(ShowStatus.RUNNING, effect);
  }

  static actorMissing(message: string = "Has anyone seen the lead?"): StageEffect {
    return new StageEffect(ShowStatus.OPENING_NIGHT, { error: message });
  }
}

class DirectorNotes {
  private store = new Map<string, unknown>();

  set<T>(key: string, value: T): void {
    this.store.set(key, value);
  }

  get<T>(key: string): T | undefined {
    return this.store.get(key) as T | undefined;
  }
}

class ScriptManager<T> implements Producible {
  private scripts = new Map<string, PropStorage<T>>();

  get(key: string): T | undefined {
    const entry = this.scripts.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiry) {
      this.scripts.delete(key);
      return undefined;
    }
    return entry.value;
  }

  set(key: string, value: T, ttlMs: number, tags: string[] = []): void {
    this.scripts.set(key, {
      value,
      expiry: Date.now() + ttlMs,
      tags,
    });
  }

  invalidateByTag(tag: string): number {
    let count = 0;
    for (const [key, entry] of this.scripts) {
      if (entry.tags.includes(tag)) {
        this.scripts.delete(key);
        count++;
      }
    }
    return count;
  }

  serialize(): Buffer {
    return Buffer.from(JSON.stringify([...this.scripts]));
  }

  deserialize(data: Buffer): void {
    const parsed = JSON.parse(data.toString());
    this.scripts = new Map(parsed);
  }
}

// --- Functions ---

function createShowRundown(): Map<string, CueHandler<StageEffect>> {
  return new Map();
}

function composeSceneTransitions(...transitions: SceneTransition[]): SceneTransition {
  return (next) =>
    transitions.reduceRight((handler, transition) => transition(handler), next);
}

async function executeProductionCue(
  cue: LightingCue,
  handler: CueHandler<StageEffect>,
  transitions: SceneTransition[] = []
): Promise<StageEffect> {
  const composed = composeSceneTransitions(...transitions);
  const finalHandler = composed(handler);
  const notes = new DirectorNotes();
  return finalHandler(cue, notes);
}

function isStandingOvation(result: CueResult<unknown>): result is { ok: true; value: unknown } {
  return result.ok;
}

async function retakeScene<T>(
  fn: () => Promise<T>,
  attempts: number = 3,
  delayMs: number = 1000
): Promise<CueResult<T>> {
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
  return { ok: false, error: new Error("Director called it — no more retakes") };
}

export {
  type StageDirection,
  type CueHandler,
  type CueResult,
  LightingIntensity,
  ShowStatus,
  LightingCue,
  StageEffect,
  DirectorNotes,
  ScriptManager,
  createShowRundown,
  executeProductionCue,
  retakeScene,
};
