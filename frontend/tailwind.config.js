/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy:     "#0f1117",
        card:     "#1a1d27",
        elevated: "#252836",
        dim:      "#2a2d3d",
        muted:    "#6272a4",
        prose:    "#c8ccd8",
        accent:   "#4c9be8",
        gain:     "#50fa7b",
        loss:     "#ff5555",
        warn:     "#ffb86c",
        purple:   "#bd93f9",
        cyan:     "#8be9fd",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};

