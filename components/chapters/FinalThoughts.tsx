"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { SmogParticles } from "@/viz/SmogParticles";
import { CHAPTER_CONTENT } from "@/data/content";
import { motion } from "framer-motion";
import { Layers, Cpu, ShieldCheck, Eye } from "lucide-react";

export const FinalThoughts: React.FC = () => {
  const content = CHAPTER_CONTENT.final;
  const icons = [Layers, Cpu, ShieldCheck, Eye];

  return (
    <section id="chapter-final" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background/90 overflow-hidden">
      {/* Light Clear Air Particles */}
      <SmogParticles density={0.15} speedMultiplier={0.5} />

      <div className="relative z-10 max-w-5xl mx-auto space-y-12 text-center">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
          centered
        />

        {/* 4 Pillars Converging */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-4">
          {content.pillars.map((pillar, idx) => {
            const Icon = icons[idx];
            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.15 }}
                className="p-6 rounded-2xl bg-surface/80 border border-amber-500/20 backdrop-blur-md flex flex-col items-center text-center space-y-3"
              >
                <div className="w-12 h-12 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center">
                  <Icon className="w-6 h-6" />
                </div>
                <div className="font-bold text-white text-base">{pillar.title}</div>
                <div className="text-xs text-gray-400">{pillar.desc}</div>
              </motion.div>
            );
          })}
        </div>

        {/* Full-Circle Conclusion Message */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="pt-12 max-w-2xl mx-auto space-y-4"
        >
          <div className="h-0.5 w-24 bg-gradient-to-r from-transparent via-amber-500 to-transparent mx-auto mb-6" />
          <p className="text-gray-300 text-lg leading-relaxed font-medium">
            By unifying rigorous preprocessing, gradient boosted ensembles, resampled statistical testing, and SHAP explainability, machine learning empowers evidence-based environmental policy for healthier Indian cities.
          </p>
          <div className="text-xs font-mono text-amber-400 pt-4 uppercase tracking-widest">
            AQI Data Science Story Completed
          </div>
        </motion.div>
      </div>
    </section>
  );
};
