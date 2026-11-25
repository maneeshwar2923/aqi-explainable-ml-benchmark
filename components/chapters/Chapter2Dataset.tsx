"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { StatCounter } from "@/ui/StatCounter";
import { CityMap } from "@/viz/CityMap";
import { CHAPTER_CONTENT } from "@/data/content";

export const Chapter2Dataset: React.FC = () => {
  const content = CHAPTER_CONTENT.dataset;

  return (
    <section id="chapter-dataset" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background/90">
      <div className="max-w-6xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        {/* 4 Stat Counters */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {content.stats.map((s, idx) => (
            <StatCounter
              key={idx}
              value={s.value}
              label={s.label}
              suffix={s.suffix}
            />
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          {/* 26-City India Map */}
          <CityMap />

          {/* Pollutant Feature Grid */}
          <div className="space-y-4">
            <div className="text-xs font-mono text-amber-400 uppercase tracking-widest">
              8 Monitored Pollutant Sub-Indices
            </div>
            <div className="grid grid-cols-2 gap-3">
              {content.pollutants.map((p, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl bg-surface/60 border border-surface-border hover:border-amber-500/30 transition-all"
                >
                  <div className="font-mono text-base font-bold text-white">{p.name}</div>
                  <div className="text-xs text-gray-400 line-clamp-1">{p.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
