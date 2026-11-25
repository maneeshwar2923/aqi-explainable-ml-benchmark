export const AQI_COLORS = {
  good: { hex: "#00E400", label: "Good (0–50)", text: "Air quality is satisfactory and poses little or no risk." },
  satisfactory: { hex: "#55FF00", label: "Satisfactory (51–100)", text: "Minor breathing discomfort to sensitive people." },
  moderate: { hex: "#FFFF00", label: "Moderate (101–200)", text: "Breathing discomfort to people with lungs, asthma, and heart diseases." },
  poor: { hex: "#FF7E00", label: "Poor (201–300)", text: "Breathing discomfort to most people on prolonged exposure." },
  veryPoor: { hex: "#FF0000", label: "Very Poor (301–400)", text: "Respiratory illness on prolonged exposure." },
  severe: { hex: "#99004C", label: "Severe (401–500)", text: "Affects healthy people and seriously impacts those with existing diseases." },
} as const;

export const PALETTE = {
  bg: "#080a0f",
  surface: "#111622",
  surfaceBorder: "#1f293d",
  textPrimary: "#f3f4f6",
  textMuted: "#9ca3af",
  accentOrange: "#f97316",
  accentRed: "#ef4444",
  accentGold: "#eab308",
  accentBlue: "#3b82f6",
} as const;
