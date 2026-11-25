"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { PipelineFlow } from "@/viz/PipelineFlow";
import { CHAPTER_CONTENT } from "@/data/content";

export const Chapter3Preprocessing: React.FC = () => {
  const content = CHAPTER_CONTENT.preprocessing;

  return (
    <section id="chapter-preprocessing" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background/95">
      <div className="max-w-6xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <PipelineFlow />
      </div>
    </section>
  );
};
