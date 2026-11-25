"use client";

import React from "react";
import { ChapterHeader } from "@/ui/ChapterHeader";
import { CHAPTER_CONTENT } from "@/data/content";
import { motion } from "framer-motion";
import { Zap, Flame, Calendar, MapPin, CheckCircle2, Award } from "lucide-react";

export const Chapter8Insights: React.FC = () => {
  const content = CHAPTER_CONTENT.insights;
  const icons = [Zap, Flame, Calendar, MapPin, CheckCircle2, Award];

  return (
    <section id="chapter-insights" className="relative min-h-screen py-24 px-6 flex flex-col justify-center bg-background">
      <div className="max-w-6xl mx-auto space-y-12 w-full">
        <ChapterHeader
          badge={content.badge}
          title={content.title}
          subtitle={content.subtitle}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {content.cards.map((card, idx) => {
            const Icon = icons[idx];
            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                className="p-6 rounded-3xl bg-surface/70 border border-surface-border backdrop-blur-md hover:border-amber-500/40 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-center mb-4">
                    <Icon className="w-6 h-6" />
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">{card.title}</h4>
                  <p className="text-gray-300 text-sm leading-relaxed">{card.text}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
