import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#080a0f",
        foreground: "#f3f4f6",
        surface: "#111622",
        "surface-border": "#1f293d",
        aqi: {
          good: "#00E400",
          satisfactory: "#55FF00",
          moderate: "#FFFF00",
          poor: "#FF7E00",
          veryPoor: "#FF0000",
          severe: "#99004C",
        },
      },
      fontFamily: {
        sans: ["var(--font-space-grotesk)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
