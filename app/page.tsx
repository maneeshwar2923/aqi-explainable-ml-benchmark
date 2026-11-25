"use client";

import React, { useEffect } from "react";
import { initLenis } from "@/lib/lenis";
import { ScrollTrigger } from "@/lib/gsapConfig";
import { useStoryStore } from "@/store/useStoryStore";
import { ScrollProgressRail } from "@/ui/ScrollProgressRail";

import { Chapter1Problem } from "@/chapters/Chapter1Problem";
import { Chapter2Dataset } from "@/chapters/Chapter2Dataset";
import { Chapter3Preprocessing } from "@/chapters/Chapter3Preprocessing";
import { Chapter4Models } from "@/chapters/Chapter4Models";
import { Chapter5Comparison } from "@/chapters/Chapter5Comparison";
import { Chapter6Significance } from "@/chapters/Chapter6Significance";
import { Chapter7SHAP } from "@/chapters/Chapter7SHAP";
import { Chapter8Insights } from "@/chapters/Chapter8Insights";
import { FinalThoughts } from "@/chapters/FinalThoughts";

export default function Home() {
  const { setCurrentChapter, setScrollProgress } = useStoryStore();

  useEffect(() => {
    const lenis = initLenis();

    const chapterIds = [
      "problem", "dataset", "preprocessing", "models",
      "comparison", "significance", "shap", "insights", "final"
    ];

    chapterIds.forEach((id, index) => {
      ScrollTrigger.create({
        trigger: `#chapter-${id}`,
        start: "top 60%",
        end: "bottom 60%",
        onEnter: () => setCurrentChapter(index + 1),
        onEnterBack: () => setCurrentChapter(index + 1),
      });
    });

    ScrollTrigger.create({
      start: 0,
      end: "max",
      onUpdate: (self) => {
        setScrollProgress(self.progress);
      },
    });

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
      lenis?.destroy();
    };
  }, [setCurrentChapter, setScrollProgress]);

  return (
    <main className="relative min-h-screen bg-background text-foreground grid-pattern overflow-hidden">
      <ScrollProgressRail />

      <Chapter1Problem />
      <Chapter2Dataset />
      <Chapter3Preprocessing />
      <Chapter4Models />
      <Chapter5Comparison />
      <Chapter6Significance />
      <Chapter7SHAP />
      <Chapter8Insights />
      <FinalThoughts />
    </main>
  );
}
