"use client";

import React from "react";
import { motion } from "framer-motion";

interface ChapterHeaderProps {
  badge: string;
  title: string;
  subtitle: string;
  centered?: boolean;
}

export const ChapterHeader: React.FC<ChapterHeaderProps> = ({
  badge,
  title,
  subtitle,
  centered = false,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className={`mb-12 ${centered ? "text-center" : "text-left"}`}
    >
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-mono tracking-widest uppercase mb-4">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
        {badge}
      </div>
      <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
        {title}
      </h2>
      <p className="text-lg sm:text-xl text-gray-400 max-w-2xl leading-relaxed">
        {subtitle}
      </p>
    </motion.div>
  );
};
