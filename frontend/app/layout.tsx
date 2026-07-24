import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StockLens — AI Stock Price Prediction",
  description:
    "Stock price prediction using Bidirectional LSTM with walk-forward validation, " +
    "XGBoost comparison, MC-Dropout uncertainty quantification, and live market data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var t = localStorage.getItem('theme') ||
                  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
                if (t === 'dark') document.documentElement.classList.add('dark');
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
        {children}
      </body>
    </html>
  );
}
