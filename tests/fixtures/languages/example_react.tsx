/**
 * Example React/TSX file exercising all extractable element types.
 *
 * This fixture ensures the magaldi index always contains React-specific
 * elements (components, hooks, memo/forwardRef wrappers) when parsing its own repo.
 */

import React, { useState, useEffect, useCallback, useRef, forwardRef, memo } from "react";

// Constants
const MAX_ITEMS = 50;
const DEBOUNCE_MS = 300;
const THEME_COLORS = {
  primary: "#3b82f6",
  secondary: "#64748b",
  danger: "#ef4444",
} as const;

// Type aliases
type Theme = keyof typeof THEME_COLORS;
type ItemId = string | number;

// Interfaces
interface ListItem {
  id: ItemId;
  title: string;
  description?: string;
  status: "active" | "archived";
}

interface ListProps {
  items: ListItem[];
  onSelect: (item: ListItem) => void;
  maxVisible?: number;
  theme?: Theme;
}

interface InputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

// Enums
enum SortOrder {
  ASC = "asc",
  DESC = "desc",
}

// --- Custom Hooks ---

function useDebounce<T>(value: T, delayMs: number = DEBOUNCE_MS): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debouncedValue;
}

function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
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

function Badge({ text, theme = "primary" }: { text: string; theme?: Theme }) {
  return (
    <span style={{ backgroundColor: THEME_COLORS[theme], color: "white", padding: "2px 8px", borderRadius: "4px" }}>
      {text}
    </span>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div style={{ textAlign: "center", padding: "2rem", color: "#64748b" }}>
      <p>{message}</p>
    </div>
  );
}

function ItemCard({ item, onSelect }: { item: ListItem; onSelect: (item: ListItem) => void }) {
  return (
    <div onClick={() => onSelect(item)} style={{ padding: "1rem", border: "1px solid #e2e8f0", cursor: "pointer" }}>
      <h3>{item.title}</h3>
      {item.description && <p>{item.description}</p>}
      <Badge text={item.status} theme={item.status === "active" ? "primary" : "secondary"} />
    </div>
  );
}

// React.memo component
const MemoizedItemCard = memo(ItemCard);

// forwardRef component
const SearchInput = forwardRef<HTMLInputElement, InputProps>(function SearchInput(
  { value, onChange, placeholder = "Search...", disabled = false },
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

// Main component with hooks
function FilterableList({ items, onSelect, maxVisible = MAX_ITEMS, theme = "primary" }: ListProps) {
  const [query, setQuery] = useState("");
  const [sortOrder, setSortOrder] = useLocalStorage<SortOrder>("sort-order", SortOrder.ASC);
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedQuery = useDebounce(query);

  const filteredItems = items
    .filter((item) => item.title.toLowerCase().includes(debouncedQuery.toLowerCase()))
    .sort((a, b) => {
      const cmp = a.title.localeCompare(b.title);
      return sortOrder === SortOrder.ASC ? cmp : -cmp;
    })
    .slice(0, maxVisible);

  const handleClear = useCallback(() => {
    setQuery("");
    inputRef.current?.focus();
  }, []);

  if (items.length === 0) {
    return <EmptyState message="No items available" />;
  }

  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        <SearchInput ref={inputRef} value={query} onChange={setQuery} />
        <button onClick={handleClear}>Clear</button>
        <button onClick={() => setSortOrder(sortOrder === SortOrder.ASC ? SortOrder.DESC : SortOrder.ASC)}>
          Sort {sortOrder === SortOrder.ASC ? "↓" : "↑"}
        </button>
      </div>
      <div>
        {filteredItems.map((item) => (
          <MemoizedItemCard key={item.id} item={item} onSelect={onSelect} />
        ))}
      </div>
    </div>
  );
}

// Arrow function component
const StatusIndicator = ({ active }: { active: boolean }) => (
  <span style={{ color: active ? THEME_COLORS.primary : THEME_COLORS.secondary }}>
    {active ? "●" : "○"}
  </span>
);

export { FilterableList, SearchInput, Badge, StatusIndicator, useDebounce, useLocalStorage };
export type { ListItem, ListProps, InputProps, Theme };
