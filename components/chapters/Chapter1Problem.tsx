"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { SmogParticles } from "@/viz/SmogParticles";
import { RevealText } from "@/ui/RevealText";
import { CHAPTER_CONTENT } from "@/data/content";
import { Activity, ShieldAlert, BellRing, Cpu } from "lucide-react";

export const Chapter1Problem: React.FC = () => {
  const content = CHAPTER_CONTENT.problem;
  const iconMap = [Activity, ShieldAlert, BellRing, Cpu];

  return (
    <section id="chapter-problem" className="relative min-h-screen py-24 px-6 flex flex-col justify-center overflow-hidden">
      {/* Background Smog Canvas */}
      <SmogParticles density={0.7} speedMultiplier={1.2} />

      <div className="relative z-10 max-w-5xl mx-auto space-y-12">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <RevealText
          text="Air pollution across major Indian urban centers poses severe public health threats. Precise predictive modeling transforms raw sensor measurements into proactive environmental defense."
          className="text-xl sm:text-2xl font-medium text-gray-200 leading-relaxed max-w-3xl"
        />

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-6">
          {content.bullets.map((b, idx) => {
            const Icon = iconMap[idx];
            return (
              <div
                key={b.id}
                className="p-5 rounded-2xl bg-surface/60 border border-surface-border backdrop-blur-md flex items-start gap-4 hover:border-amber-500/40 transition-all"
              >
                <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-center shrink-0">
                  <Icon className="w-5 h-5" />
                </div>
                <p className="text-gray-300 text-sm sm:text-base leading-relaxed">
                  {b.text}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
