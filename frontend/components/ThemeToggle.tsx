"use client";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = saved ? saved === "dark" : prefersDark;
    setDark(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  return (
    <motion.button
      onClick={toggle}
      whileTap={{ scale: 0.9 }}
      whileHover={{ scale: 1.05 }}
      className="relative flex items-center justify-center w-10 h-10 rounded-xl
                 border border-[var(--border)] bg-[var(--surface)]
                 hover:border-[var(--primary)] transition-colors duration-200"
      aria-label="Toggle theme"
    >
      <motion.div
        key={dark ? "moon" : "sun"}
        initial={{ opacity: 0, rotate: -90, scale: 0.5 }}
        animate={{ opacity: 1, rotate: 0,   scale: 1 }}
        exit={{    opacity: 0, rotate:  90, scale: 0.5 }}
        transition={{ duration: 0.25 }}
      >
        {dark
          ? <Sun  size={17} className="text-[var(--accent)]" />
          : <Moon size={17} className="text-[var(--primary)]" />
        }
      </motion.div>
    </motion.button>
  );
}
