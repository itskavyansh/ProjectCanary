/**
 * Rendering-layer text sanitiser for the /simulation section.
 * Converts the LaTeX fragments that appear in exported simulation JSON into
 * plain text. The exported JSON files themselves are never modified.
 */

export function sanitizeSimText(input: string): string {
  let s = input;
  // Implicit multiplication before a control word: 2\pi -> 2*\pi, )\sqrt -> )*\sqrt
  s = s.replace(/(\d|\))\s*\\(?=[A-Za-z])/g, "$1*\\");
  // \sqrt{...} -> sqrt(...)
  s = s.replace(/\\sqrt\s*\{([^{}]*)\}/g, "sqrt($1)");
  // \frac{a}{b} -> (a)/(b)
  s = s.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
  // Operators with symbols
  s = s.replace(/\\cdot|\\times/g, "*");
  s = s.replace(/\\approx/g, "~");
  s = s.replace(/\\leq?|\\le\b/g, "<=").replace(/\\geq?|\\ge\b/g, ">=");
  s = s.replace(/\\pm/g, "+/-");
  // Subscripts: S_{\max} -> S_max, X_{abc} -> X_abc
  s = s.replace(/_\{\s*\\?([A-Za-z0-9]+)\s*\}/g, "_$1");
  // Remaining control words: \tan -> tan, \sigma -> sigma, ...
  s = s.replace(/\\([A-Za-z]+)/g, "$1");
  // \left( \right) and escaped punctuation already handled above; drop leftovers
  s = s.replace(/\\[,;:!]/g, " ");
  // Strip math delimiters
  s = s.replace(/\$\$?/g, "");
  // Any stray backslash
  s = s.replace(/\\/g, "");
  return s;
}

export function deepSanitizeSim<T>(value: T): T {
  if (typeof value === "string") return sanitizeSimText(value) as unknown as T;
  if (Array.isArray(value)) {
    return value.map((v) => deepSanitizeSim(v)) as unknown as T;
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = deepSanitizeSim(v);
    }
    return out as T;
  }
  return value;
}
