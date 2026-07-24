/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Autumn Day Mode ──────────────────────────────────────
        autumn: {
          bg:        "#F5F0E8",   // warm beige
          surface:   "#EDE5D0",   // card beige
          border:    "#D4C8A8",   // soft tan border
          primary:   "#6B7C3A",   // olive green
          secondary: "#8B3A3A",   // maroon brown
          accent:    "#C9A84C",   // mustard yellow
          text:      "#2C1810",   // dark brown
          muted:     "#7A6A5A",   // warm grey
          subtle:    "#B8A898",   // lighter muted
          success:   "#4A7A3A",   // forest green
          danger:    "#A03030",   // deep red
        },
        // ── Royal Night Mode ─────────────────────────────────────
        royal: {
          bg:        "#0F0A1E",   // deep royal purple-black
          surface:   "#1A1130",   // card dark royal
          surface2:  "#221840",   // slightly lighter surface
          border:    "#2D2050",   // royal border
          primary:   "#8B7FC4",   // lavender-royal accent
          secondary: "#C4A882",   // warm gold
          accent:    "#E8D5A3",   // light parchment (numbers)
          text:      "#EDE8F5",   // light lavender-white
          muted:     "#9B94B8",   // muted royal
          subtle:    "#6B6490",   // darker muted
          success:   "#6BCB8B",   // mint green
          danger:    "#F08080",   // soft coral
        },
      },
      fontFamily: {
        sans:  ["Inter", "system-ui", "sans-serif"],
        mono:  ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        "autumn-card": "0 2px 16px rgba(44,24,16,0.08), 0 1px 4px rgba(44,24,16,0.06)",
        "royal-card":  "0 2px 24px rgba(15,10,30,0.6), 0 1px 6px rgba(139,127,196,0.1)",
        "glow-olive":  "0 0 20px rgba(107,124,58,0.25)",
        "glow-royal":  "0 0 20px rgba(139,127,196,0.25)",
      },
      backgroundImage: {
        "autumn-hero": "radial-gradient(ellipse at 20% 0%, rgba(107,124,58,0.12) 0%, transparent 60%), radial-gradient(ellipse at 80% 100%, rgba(201,168,76,0.10) 0%, transparent 60%)",
        "royal-hero":  "radial-gradient(ellipse at 20% 0%, rgba(139,127,196,0.12) 0%, transparent 60%), radial-gradient(ellipse at 80% 100%, rgba(196,168,130,0.08) 0%, transparent 60%)",
      },
      animation: {
        "fade-up":    "fadeUp 0.5s ease-out forwards",
        "fade-in":    "fadeIn 0.4s ease-out forwards",
        "shimmer":    "shimmer 1.8s infinite",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)"    },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0"  },
        },
      },
    },
  },
  plugins: [],
};
