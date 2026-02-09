# Teatro Magaldi - Narrative Fixture Files

## Concept

Replace the 7 generic `example_*` fixture files with a **cohesive narrative** set in a **1930s Buenos Aires theater** -- named after Agustín Magaldi, the project's namesake. Each language implements a different "module" of the same fictional system.

The domain (shows, performers, audiences, tickets, backstage chaos) naturally requires every element type magaldi extracts, while being memorable and fun.

## Module Map

| Language | File | Domain Slice |
|----------|------|-------------|
| **Python** | `teatro_performers.py` | Performer management -- artists, roles, rehearsals, diva tantrums |
| **JavaScript** | `teatro_ticketing.js` | Ticket sales -- box office, scalpers, dynamic pricing, sold-out chaos |
| **TypeScript** | `teatro_production.ts` | Show production -- scripts, sets, lighting cues, costume changes |
| **React/TSX** | `teatro_marquee.tsx` | The theater marquee UI -- show listings, reviews, star ratings |
| **PHP** | `teatro_backstage.php` | Backstage ops -- props inventory, stage crew, wardrobe, incident reports |
| **Rust** | `teatro_orchestra.rs` | Orchestra pit -- musicians, instruments, sheet music, tuning, tempo |
| **Bash** | `teatro_stagehand.sh` | Stage operations -- curtain control, spotlights, smoke machines |

## Recurring Characters (cross-language threads)

These appear across multiple files, making cross-language search interesting:

- **Agustín Magaldi** -- legendary performer (Python), on ticket stubs (JS), on the marquee (TSX), in the orchestra program (Rust)
- **"La Noche Estrellada"** -- the flagship show being produced (TS), sold out (JS), reviewed (TSX), rehearsed (Python)
- **The Phantom Scalper** -- ticketing antagonist (JS), backstage security concern (PHP), referenced in stagehand logs (Bash)
- **La Divina** -- temperamental diva performer (Python), costume changes break records (PHP), spotlight demands (Bash)
- **The Old Spotlight** -- needs constant maintenance (Bash), temperamental like the diva, referenced in production notes (TS)

## Element Coverage Per File

Every element type from the current `example_*` files must be preserved 1:1 in the teatro equivalent. The narrative must be a **skin**, not a reduction.

### Python: `teatro_performers.py`

Current `example_python.py` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `MAX_RETRIES` | constant | `MAX_ENCORES` |
| `DEFAULT_TIMEOUT` | constant | `DEFAULT_REHEARSAL_HOURS` |
| `SUPPORTED_FORMATS` | constant | `SUPPORTED_ACT_TYPES` |
| `logger` | variable | `stage_manager_log` |
| `_registry` | variable (private, typed) | `_performer_registry` |
| `_initialized` | variable (private) | `_auditions_open` |
| `Color` | enum (auto) | `ActType` (COMEDY, DRAMA, MUSICAL) |
| `Priority` | enum (int values) | `VocalRange` (SOPRANO=1, ALTO=2, TENOR=3, BASS=4) |
| `Serializable` | protocol | `Performable` (methods: `perform`, `take_bow`) |
| `Config` | dataclass | `PerformerProfile` (name, vocal_range, years_active, roles) |
| `BaseProcessor` | class | `BasePerformer` |
| `TextProcessor` | class (inherits) | `Diva` (inherits BasePerformer) -- overrides everything |
| `parse_config` | function | `parse_audition_results` |
| `register_processor` | function | `register_performer` |
| `fetch_remote_config` | async function | `fetch_touring_schedule` |
| `process_batch` | async function | `rehearse_ensemble` |
| `with_retry` | decorator definition | `@requires_rehearsal` |
| `unreliable_operation` | decorated function | `attempt_high_note` (decorated with @requires_rehearsal) |
| `create_default` | staticmethod | `create_understudy` (staticmethod on BasePerformer) |

### JavaScript: `teatro_ticketing.js`

Current `example_javascript.js` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `MAX_BUFFER_SIZE` | constant | `MAX_SEATS` |
| `DEFAULT_ENCODING` | constant | `DEFAULT_CURRENCY` |
| `RETRY_DELAYS` | constant (array) | `INTERMISSION_PRICES` (array of price tiers) |
| `connectionCount` | variable (let) | `ticketsSold` |
| `isInitialized` | variable (let) | `boxOfficeOpen` |
| `EventHandler` | class | `BoxOffice` |
| `ConnectionPool` | class (extends) | `OnlineBookingPortal` (extends BoxOffice) |
| `parseConfig` | function | `parseSeatChart` |
| `validateInput` | function | `validateTicket` |
| `fetchWithRetry` | async function | `processOnlineBooking` |
| `processParallel` | async function | `processGroupReservation` (default param `concurrency = 4`) |
| `debounce` | arrow function (higher-order) | `rateLimitScalper` |
| `memoize` | arrow function (higher-order) | `cacheSeatPricing` |
| `get size` | getter | `get availableSeats` |

### TypeScript: `teatro_production.ts`

Current `example_typescript.ts` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `API_VERSION` | constant | `SHOW_SEASON` |
| `MAX_PAGE_SIZE` | constant | `MAX_SCENE_COUNT` |
| `ALLOWED_METHODS` | constant (as const) | `STAGE_DIRECTIONS` (as const) |
| `requestCount` | variable (let) | `sceneChangeCount` |
| `HttpMethod` | type alias (indexed) | `StageDirection` |
| `Handler<T>` | type alias (generic fn) | `CueHandler<T>` |
| `Middleware` | type alias (higher-order fn) | `SceneTransition` |
| `Result<T, E>` | type alias (discriminated union) | `CueResult<T, E>` |
| `DeepPartial<T>` | type alias (mapped/conditional) | `DraftScript<T>` |
| `LogLevel` | enum (string) | `LightingIntensity` (DIM, NORMAL, BRIGHT, SPOTLIGHT) |
| `StatusCode` | enum (numeric) | `ShowStatus` (REHEARSAL=100, DRESS_REHEARSAL=200, ...) |
| `Direction` | const enum | `CurtainState` (const enum: OPEN, CLOSED, RISING, FALLING) |
| `Serializable` | interface | `Producible` |
| `Repository<T>` | interface (generic) | `ScriptLibrary<T>` |
| `QueryOptions` | interface (optional props) | `CastingCriteria` |
| `Logger` | interface | `StageLog` |
| `CacheEntry<T>` | interface (generic) | `PropStorage<T>` |
| `Request` | class (public params) | `LightingCue` |
| `Response` | class (static methods) | `StageEffect` |
| `Context` | class (generic methods) | `DirectorNotes` |
| `InMemoryCache<T>` | class (generic, implements) | `ScriptManager<T>` (implements Producible) |
| `createRouter` | function | `createShowRundown` |
| `composeMiddleware` | function (rest param) | `composeSceneTransitions` |
| `handleRequest` | async function | `executeProductionCue` |
| `isSuccess` | type predicate | `isStandingOvation` |
| `retry<T>` | async generic function | `retakeScene<T>` |

### React/TSX: `teatro_marquee.tsx`

Current `example_react.tsx` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `MAX_ITEMS` | constant | `MAX_SHOWS_DISPLAYED` |
| `DEBOUNCE_MS` | constant | `MARQUEE_SCROLL_MS` |
| `THEME_COLORS` | constant (as const) | `MARQUEE_THEMES` (as const: art_deco, noir, golden_age) |
| `Theme` | type alias (keyof) | `MarqueeTheme` |
| `ItemId` | type alias (union) | `ShowId` |
| `ListItem` | interface | `ShowListing` |
| `ListProps` | interface | `MarqueeProps` |
| `InputProps` | interface | `SearchBoxProps` |
| `SortOrder` | enum (string) | `SortShows` (BY_DATE, BY_RATING) |
| `useDebounce<T>` | custom hook (generic) | `useMarqueeScroll<T>` |
| `useLocalStorage<T>` | custom hook (generic) | `useFavoriteShows<T>` |
| `Badge` | function component | `StarRating` |
| `EmptyState` | function component | `NoShowsTonight` |
| `ItemCard` | function component | `ShowCard` |
| `FilterableList` | main component | `MarqueeBanner` (main) |
| `MemoizedItemCard` | memo-wrapped | `MemoizedShowCard` |
| `SearchInput` | forwardRef | `ShowSearchInput` (forwardRef) |
| `StatusIndicator` | arrow component | `SoldOutBadge` |

### PHP: `teatro_backstage.php`

Current `example_php.php` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `App\Example` | namespace | `Teatro\Backstage` |
| `MAX_CONNECTIONS` | constant | `MAX_CREW_MEMBERS` |
| `DEFAULT_TIMEOUT` | constant | `DEFAULT_SHIFT_HOURS` |
| `SUPPORTED_DRIVERS` | constant (array) | `SUPPORTED_DEPARTMENTS` |
| `Status` | enum (string-backed) | `PropCondition` (PRISTINE, WORN, DAMAGED, MISSING) |
| `Priority` | enum (int-backed) | `IncidentSeverity` (MINOR=1, MODERATE=5, MAJOR=10, CATASTROPHIC=100) |
| `Repository` | interface | `PropInventory` |
| `Transformer` | interface | `CostumeAlterations` |
| `EventSubscriber` | interface (static method) | `IncidentResponder` |
| `HasTimestamps` | trait | `HasMaintenanceLog` |
| `HasSlug` | trait | `HasInventoryTag` |
| `Loggable` | trait | `BackstageLoggable` |
| `Route` | class (PHP attribute) | `Cue` (#[Attribute]) |
| `Validate` | class (PHP attribute) | `SafetyCheck` (#[Attribute]) |
| `BaseEntity` | abstract class | `AbstractStageProp` (abstract, uses HasMaintenanceLog) |
| `User` | class (extends, implements, uses traits) | `WardrobeItem` (extends AbstractStageProp, implements IncidentResponder, uses HasInventoryTag, BackstageLoggable) |
| `UserDTO` | final readonly class | `IncidentReport` (final readonly) |
| `UserRepository` | class (implements) | `PropRoom` (implements PropInventory) |
| `createUser` | function | `registerProp` |
| `validateEmail` | function | `validateSafetyInspection` |
| `listUsers` | function (with #[Route]) | `listIncidents` (with #[Cue('/backstage/incidents', method: 'GET')]) |

### Rust: `teatro_orchestra.rs`

Current `example_rust.rs` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `MAX_RETRIES` | constant | `MAX_TUNING_ATTEMPTS` |
| `DEFAULT_TIMEOUT_SECS` | constant | `DEFAULT_TEMPO_BPM` |
| `BUFFER_SIZE` | constant | `ORCHESTRA_PIT_CAPACITY` |
| `GLOBAL_COUNTER` | static (AtomicU64) | `PERFORMANCE_COUNTER` |
| `Status` | enum (struct variant) | `TuningStatus` (InTune, Flat, Sharp { cents: f32 }, Broken) |
| `Priority` | enum (discriminant) | `SectionPriority` (Strings=1, Woodwinds=5, Brass=10, Percussion=100) |
| `AppError` | enum (tuple/struct variants) | `OrchestraError` (WrongNote(String), OutOfTune(String), BrokenInstrument(Box<dyn Error>), Timeout { elapsed, limit }) |
| `Repository<T>` | trait (generic) | `MusicLibrary<T>` |
| `Serializable` | trait | `Tunable` (tune, check_tuning) |
| `Transformer` | trait (assoc types) | `Transcriber` (assoc types: Input, Output) |
| `EventHandler` | trait (supertraits) | `ConductorSignal` (Send + Sync) |
| `Config` | struct (derive) | `Instrument` (name, section, tuning_frequency, is_rented) |
| `User` | struct | `Musician` (id, name, instrument, section, status) |
| `InMemoryRepo<T>` | struct (generic) | `SheetMusicArchive<T>` |
| `Cache<V>` | struct (generic) | `RehearsalSchedule<V>` |
| `CacheEntry<V>` | struct (generic) | `RehearsalSlot<V>` |
| `Config::new` | impl method | `Instrument::new` |
| `Config::with_timeout` | builder method | `Instrument::with_tuning` |
| `Config::validate` | method (Result) | `Instrument::validate_condition` |
| `User::new` | impl method | `Musician::new` |
| `User::activate/suspend/is_active` | impl methods | `Musician::tune_up/rest/is_ready` |
| `Display for User` | trait impl | `Display for Musician` |
| `Display for AppError` | trait impl | `Display for OrchestraError` |
| `Repository<User> for InMemoryRepo<User>` | trait impl | `MusicLibrary<Musician> for SheetMusicArchive<Musician>` |
| `parse_config` | free function | `parse_sheet_music` |
| `validate_email` | free function | `validate_instrument_serial` |
| `retry<F, T>` | generic function | `retry_tuning<F, T>` |
| `process_batch<T, F>` | generic function | `rehearse_section<T, F>` |

### Bash: `teatro_stagehand.sh`

Current `example_bash.sh` elements → Teatro equivalents:

| Current Element | Type | Teatro Equivalent |
|----------------|------|-------------------|
| `common.sh` | source (import) | `teatro_common.sh` |
| `MAX_RETRIES` | readonly constant | `MAX_CURTAIN_CALLS` |
| `DEFAULT_TIMEOUT` | readonly constant | `DEFAULT_INTERMISSION_SECS` |
| `LOG_LEVELS` | readonly array | `CUE_TYPES` ("LIGHTS" "SOUND" "CURTAIN" "SMOKE") |
| `CONFIG_DIR` | readonly | `STAGE_CONFIG_DIR` |
| `CACHE_FILE` | readonly | `CUE_SHEET_CACHE` |
| `VERBOSE` | variable | `VERBOSE` |
| `DRY_RUN` | variable | `DRESS_REHEARSAL` (love this rename) |
| `CURRENT_LOG_LEVEL` | variable | `CURRENT_CUE_TYPE` |
| `log` | function | `stage_log` |
| `debug` | function | `stage_whisper` (debug = whisper backstage) |
| `die` | function | `curtains` (die = curtains, get it?) |
| `check_dependencies` | function | `check_stage_equipment` |
| `validate_config` | function | `validate_cue_sheet` |
| `retry_command` | function | `retry_spotlight` |
| `fetch_url` | function | `fetch_prop_from_storage` |
| `process_items` | function | `run_cue_sequence` |
| `cleanup` | function (trap) | `strike_set` (theater term for teardown) |
| `parse_args` | function | `parse_stage_directions` |
| `main` | function (entry) | `showtime` (main = showtime) |

## Easter Eggs

Each file should include at least one fun easter egg:

1. **Python**: Agustín Magaldi appears as a performer with `VocalRange.TENOR` and a comment about launching careers
2. **JavaScript**: A ticket stub for "La Noche Estrellada" with seat "Palco Presidencial"
3. **TypeScript**: A lighting cue called "The Magaldi Spotlight" -- the brightest setting
4. **TSX**: A show listing with a review: "Five stars! The indexing of performances was flawless." (self-referential)
5. **PHP**: An incident report about La Divina's costume malfunction during act 3
6. **Rust**: An instrument called "The Phantom's Violin" that's always slightly out of tune
7. **Bash**: `DRESS_REHEARSAL` mode prints "This is not the real show!" and the `curtains` function says "That's all, folks!"

## Implementation Steps

### Step 1: Create Python fixture
Write `teatro_performers.py` preserving every element type from `example_python.py`. Run existing parser tests to verify extraction still works.

### Step 2: Create JavaScript fixture
Write `teatro_ticketing.js` preserving every element type from `example_javascript.js`.

### Step 3: Create TypeScript fixture
Write `teatro_production.ts` preserving every element type from `example_typescript.ts`.

### Step 4: Create React/TSX fixture
Write `teatro_marquee.tsx` preserving every element type from `example_react.tsx`.

### Step 5: Create PHP fixture
Write `teatro_backstage.php` preserving every element type from `example_php.php`.

### Step 6: Create Rust fixture
Write `teatro_orchestra.rs` preserving every element type from `example_rust.rs`.

### Step 7: Create Bash fixture
Write `teatro_stagehand.sh` preserving every element type from `example_bash.sh`.

### Step 8: Update test references
Update all tests that reference `example_*` files to point to `teatro_*` files. Remove old `example_*` files.

### Step 9: Verify
Run full test suite. Every parser test must pass with the new fixtures. No element type regressions.

## Constraints

1. **1:1 element parity** -- every element type in the current file must exist in the teatro version
2. **Syntactically valid** -- all files must parse without errors
3. **No test regressions** -- all existing parser/extractor tests must pass
4. **Cross-references** -- recurring characters must appear in at least 2-3 files each
5. **Comments welcome** -- use comments to add flavor, but keep code structure realistic

## Success Criteria

- [ ] All 7 fixture files replaced with teatro narrative versions
- [ ] `make test-fast` passes with zero regressions
- [ ] Searching "Magaldi" in the indexed fixtures finds results in 3+ languages
- [ ] Searching "La Noche Estrellada" finds results in 3+ languages
- [ ] Every element type from the cross-file summary table is covered
- [ ] At least one easter egg per file
