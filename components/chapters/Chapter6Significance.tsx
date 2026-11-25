"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { SignificanceMeter } from "@/viz/SignificanceMeter";
import { CHAPTER_CONTENT } from "@/data/content";

export const Chapter6Significance: React.FC = () => {
  const content = CHAPTER_CONTENT.significance;

  return (
    <section id="chapter-significance" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background">
      <div className="max-w-6xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <SignificanceMeter />
      </div>
    </section>
  );
};
