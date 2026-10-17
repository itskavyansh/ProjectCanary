/**
 * =============================================================================
 * ProjectCanary — Shared LocalStorage Keys and Migration Helpers
 * =============================================================================
 * Ensures consistent key names across index.html, React components, and adapters.
 * Gracefully reads legacy "bhoomi_" keys if the new "projectcanary_" key is not yet set.
 */

export const STORAGE_KEYS = {
  THEME: "projectcanary_theme",
  LEGACY_THEME: "bhoomi_theme",

  FONT_SIZE: "projectcanary_font_size",
  LEGACY_FONT_SIZE: "bhoomi_font_size",

  HIGH_CONTRAST: "projectcanary_high_contrast",
  LEGACY_HIGH_CONTRAST: "bhoomi_high_contrast",

  NOTIFICATION_ACK: "projectcanary_ack",
  LEGACY_NOTIFICATION_ACK: "bhoomi_ack",

  BASELINE_PREFIX: "projectcanary_baseline_",
  LEGACY_BASELINE_PREFIX: "bhoomi_baseline_",
} as const;

/**
 * Reads from primary key, falling back to legacy key if null/empty.
 */
export function getStorageItemWithFallback(primaryKey: string, legacyKey: string): string | null {
  if (typeof window === "undefined" || !window.localStorage) return null;
  try {
    const val = window.localStorage.getItem(primaryKey);
    if (val !== null) return val;
    return window.localStorage.getItem(legacyKey);
  } catch {
    return null;
  }
}

/**
 * Sets primary key in localStorage without deleting legacy key.
 */
export function setStorageItem(primaryKey: string, value: string): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    window.localStorage.setItem(primaryKey, value);
  } catch {
    /* ignore storage quota or private-browsing restrictions */
  }
}
