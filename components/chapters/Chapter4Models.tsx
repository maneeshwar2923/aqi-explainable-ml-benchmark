"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { ModelCard } from "@/viz/ModelCard";
import { CHAPTER_CONTENT } from "@/data/content";
import { MODELS_DATA } from "@/data/models";

export const Chapter4Models: React.FC = () => {
  const content = CHAPTER_CONTENT.models;

  return (
    <section id="chapter-models" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background">
      <div className="max-w-6xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {MODELS_DATA.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      </div>
    </section>
  );
};
