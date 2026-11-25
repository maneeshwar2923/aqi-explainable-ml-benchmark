"use client";

import React from "react";
import { CHAPTERS } from "@/data/content";
import { useStoryStore } from "@/store/useStoryStore";
import { getLenis } from "@/lib/lenis";
import { HelpCircle } from "lucide-react";

export const ScrollProgressRail: React.FC = () => {
  const { currentChapter, explainMode, toggleExplainMode } = useStoryStore();

  const handleScrollToChapter = (id: string) => {
    const lenis = getLenis();
    const el = document.getElementById(`chapter-${id}`);
    if (el) {
      if (lenis) {
        lenis.scrollTo(el, { offset: 0, duration: 1.2 });
      } else {
        el.scrollIntoView({ behavior: "smooth" });
      }
    }
  };

  return (
    <>
      {/* Top Right Explain Mode Toggle */}
      <div className="fixed top-6 right-6 z-50 flex items-center gap-3">
        <button
          onClick={toggleExplainMode}
          className={`flex items-center gap-2 px-4 py-2 rounded-full font-mono text-xs transition-all duration-300 border backdrop-blur-md ${
            explainMode
              ? "bg-amber-500 text-black border-amber-400 font-bold shadow-lg shadow-amber-500/20"
              : "bg-surface/80 text-gray-300 border-surface-border hover:border-gray-500"
          }`}
          title="Toggle Plain-Language Explain Mode"
        >
          <HelpCircle className={`w-4 h-4 ${explainMode ? "animate-spin" : ""}`} />
          <span>{explainMode ? "EXPLAIN MODE ON" : "EXPLAIN MODE"}</span>
        </button>
      </div>

      {/* Desktop Side Rail */}
      <div className="hidden lg:flex fixed right-6 top-1/2 -translate-y-1/2 z-40 flex-col items-center gap-3 bg-surface/40 backdrop-blur-md p-3 rounded-full border border-surface-border">
        {CHAPTERS.map((ch) => {
          const isActive = currentChapter === ch.index;
          return (
            <button
              key={ch.id}
              onClick={() => handleScrollToChapter(ch.id)}
              className="group relative flex items-center"
              aria-label={`Jump to Chapter ${ch.index}: ${ch.title}`}
            >
              {/* Tooltip on hover */}
              <div className="absolute right-8 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap bg-surface text-gray-200 text-xs px-2.5 py-1 rounded border border-surface-border font-mono shadow-xl">
                0{ch.index}. {ch.title}
              </div>

              {/* Node Indicator */}
              <div
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  isActive
                    ? "bg-amber-400 scale-125 glow-orange ring-4 ring-amber-500/20"
                    : "bg-gray-600 hover:bg-gray-400"
                }`}
              />
            </button>
          );
        })}
      </div>

      {/* Mobile Bottom Progress Bar */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface/90 backdrop-blur-md border-t border-surface-border px-4 py-2.5 flex items-center justify-between">
        <div className="text-xs font-mono text-amber-400">
          CH 0{currentChapter} / 09
        </div>
        <div className="flex gap-1.5">
          {CHAPTERS.map((ch) => (
            <button
              key={ch.id}
              onClick={() => handleScrollToChapter(ch.id)}
              className={`h-1.5 rounded-full transition-all ${
                currentChapter === ch.index ? "w-6 bg-amber-400" : "w-2 bg-gray-700"
              }`}
            />
          ))}
        </div>
      </div>
    </>
  );
};
