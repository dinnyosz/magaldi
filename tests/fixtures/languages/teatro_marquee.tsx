/**
 * Teatro Magaldi — Marquee UI Module.
 *
 * The theater marquee: show listings, audience reviews, star ratings,
 * and the eternal question — is La Noche Estrellada sold out again?
 *
 * "Five stars! The indexing of performances was flawless." — Self-referential Review
 */

import React, { useState, useEffect, useCallback, useRef, forwardRef, memo } from "react";

// Constants
const MAX_SHOWS_DISPLAYED = 50;
const MARQUEE_SCROLL_MS = 300;
const MARQUEE_THEMES = {
  art_deco: "#d4af37",
  noir: "#1a1a2e",
  golden_age: "#c9a227",
} as const;

// Type aliases
type MarqueeTheme = keyof typeof MARQUEE_THEMES;
type ShowId = string | number;

// Interfaces
interface ShowListing {
  id: ShowId;
  title: string;
  description?: string;
  status: "tonight" | "coming_soon";
}

interface MarqueeProps {
  shows: ShowListing[];
  onSelect: (show: ShowListing) => void;
  maxVisible?: number;
  theme?: MarqueeTheme;
}

interface SearchBoxProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

// Enums
enum SortShows {
  BY_DATE = "by_date",
  BY_RATING = "by_rating",
}

// --- Custom Hooks ---

function useMarqueeScroll<T>(value: T, delayMs: number = MARQUEE_SCROLL_MS): T {
  const [scrolledValue, setScrolledValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setScrolledValue(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return scrolledValue;
}

function useFavoriteShows<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value: T) => {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    },
    [key]
  );

  return [storedValue, setValue];
}

// --- Components ---

function StarRating({ text, theme = "art_deco" }: { text: string; theme?: MarqueeTheme }) {
  return (
    <span style={{ backgroundColor: MARQUEE_THEMES[theme], color: "white", padding: "2px 8px", borderRadius: "4px" }}>
      {text}
    </span>
  );
}

function NoShowsTonight({ message }: { message: string }) {
  return (
    <div style={{ textAlign: "center", padding: "2rem", color: "#64748b" }}>
      <p>{message}</p>
    </div>
  );
}

function ShowCard({ item, onSelect }: { item: ShowListing; onSelect: (item: ShowListing) => void }) {
  // Easter egg: "Five stars! The indexing of performances was flawless."
  return (
    <div onClick={() => onSelect(item)} style={{ padding: "1rem", border: "1px solid #e2e8f0", cursor: "pointer" }}>
      <h3>{item.title}</h3>
      {item.description && <p>{item.description}</p>}
      <StarRating text={item.status} theme={item.status === "tonight" ? "art_deco" : "noir"} />
    </div>
  );
}

// React.memo component
const MemoizedShowCard = memo(ShowCard);

// forwardRef component
const ShowSearchInput = forwardRef<HTMLInputElement, SearchBoxProps>(function ShowSearchInput(
  { value, onChange, placeholder = "Search tonight's shows...", disabled = false },
  ref
) {
  return (
    <input
      ref={ref}
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
    />
  );
});

// Main component with hooks — the grand marquee of Teatro Magaldi
function MarqueeBanner({ shows, onSelect, maxVisible = MAX_SHOWS_DISPLAYED, theme = "art_deco" }: MarqueeProps) {
  const [query, setQuery] = useState("");
  const [sortOrder, setSortOrder] = useFavoriteShows<SortShows>("marquee-sort", SortShows.BY_DATE);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrolledQuery = useMarqueeScroll(query);

  const filteredShows = shows
    .filter((show) => show.title.toLowerCase().includes(scrolledQuery.toLowerCase()))
    .sort((a, b) => {
      const cmp = a.title.localeCompare(b.title);
      return sortOrder === SortShows.BY_DATE ? cmp : -cmp;
    })
    .slice(0, maxVisible);

  const handleClear = useCallback(() => {
    setQuery("");
    inputRef.current?.focus();
  }, []);

  if (shows.length === 0) {
    return <NoShowsTonight message="No shows tonight — the teatro is dark" />;
  }

  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        <ShowSearchInput ref={inputRef} value={query} onChange={setQuery} />
        <button onClick={handleClear}>Clear</button>
        <button onClick={() => setSortOrder(sortOrder === SortShows.BY_DATE ? SortShows.BY_RATING : SortShows.BY_DATE)}>
          Sort {sortOrder === SortShows.BY_DATE ? "↓" : "↑"}
        </button>
      </div>
      <div>
        {filteredShows.map((show) => (
          <MemoizedShowCard key={show.id} item={show} onSelect={onSelect} />
        ))}
      </div>
    </div>
  );
}

// Arrow function component
const SoldOutBadge = ({ active }: { active: boolean }) => (
  <span style={{ color: active ? MARQUEE_THEMES.art_deco : MARQUEE_THEMES.noir }}>
    {active ? "●" : "○"}
  </span>
);

export { MarqueeBanner, ShowSearchInput, StarRating, SoldOutBadge, useMarqueeScroll, useFavoriteShows };
export type { ShowListing, MarqueeProps, SearchBoxProps, MarqueeTheme };
