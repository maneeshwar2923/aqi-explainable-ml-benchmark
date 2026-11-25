import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ExplainSheet } from "@/ui/ExplainSheet";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "AQI Prediction in India — An Interactive Data Story",
  description: "A cinematic longform data story walking through machine learning models, statistical validation, and SHAP interpretability for predicting urban air quality across 26 Indian cities.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`dark ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-background text-foreground antialiased selection:bg-amber-500 selection:text-black">
        {children}
        <ExplainSheet />
      </body>
    </html>
  );
}
