"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { FeatureImportanceBars } from "@/viz/FeatureImportanceBars";
import { SHAPBeeswarm } from "@/viz/SHAPBeeswarm";
import { SHAPDependencePlot } from "@/viz/SHAPDependencePlot";
import { CHAPTER_CONTENT } from "@/data/content";

export const Chapter7SHAP: React.FC = () => {
  const content = CHAPTER_CONTENT.shap;

  return (
    <section id="chapter-shap" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background/95 space-y-16">
      <div className="max-w-6xl mx-auto space-y-12 w-full">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <FeatureImportanceBars />
        <SHAPBeeswarm />
        <SHAPDependencePlot />
      </div>
    </section>
  );
};
