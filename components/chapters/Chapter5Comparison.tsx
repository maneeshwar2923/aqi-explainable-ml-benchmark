"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { MetricBarRace } from "@/viz/MetricBarRace";
import { CHAPTER_CONTENT } from "@/data/content";

export const Chapter5Comparison: React.FC = () => {
  const content = CHAPTER_CONTENT.comparison;

  return (
    <section id="chapter-comparison" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background/95">
      <div className="max-w-6xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <MetricBarRace />
      </div>
    </section>
  );
};
