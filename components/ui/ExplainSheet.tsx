"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStoryStore } from "@/store/useStoryStore";
import { EXPLANATIONS } from "@/data/content";
import { X, BookOpen } from "lucide-react";

export const ExplainSheet: React.FC = () => {
  const { activeExplanationKey, closeExplanation } = useStoryStore();

  const explanation = activeExplanationKey ? EXPLANATIONS[activeExplanationKey] : null;

  return (
    <AnimatePresence>
      {explanation && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeExplanation}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          />

          {/* Slide-up Panel */}
          <motion.div
            initial={{ y: "100%", opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: "100%", opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed bottom-0 left-0 right-0 z-50 max-w-2xl mx-auto bg-surface border-t border-surface-border p-6 sm:p-8 rounded-t-3xl shadow-2xl"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                  <BookOpen className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-xs font-mono text-amber-400 uppercase tracking-widest">
                    Plain-Language Explanation
                  </div>
                  <h3 className="text-xl font-bold text-white">
                    {explanation.title}
                  </h3>
                </div>
              </div>
              <button
                onClick={closeExplanation}
                className="p-2 rounded-full hover:bg-surface-border text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-gray-300 text-base leading-relaxed mb-6">
              {explanation.body}
            </p>

            <div className="flex justify-end">
              <button
                onClick={closeExplanation}
                className="px-5 py-2 rounded-full bg-amber-500 text-black font-semibold text-sm hover:bg-amber-400 transition-colors"
              >
                Got it
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
