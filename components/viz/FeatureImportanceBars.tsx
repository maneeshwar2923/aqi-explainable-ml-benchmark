"use client";

import React from "react";
import { motion } from "framer-motion";
import { SHAP_FEATURE_IMPORTANCE } from "@/data/models";
import { Chip } from "@/ui/Chip";
import { HelpCircle } from "lucide-react";
import { useStoryStore } from "@/store/useStoryStore";

export const FeatureImportanceBars: React.FC = () => {
  const { openExplanation } = useStoryStore();

  const maxVal = 50;

  return (
    <div className="w-full max-w-4xl mx-auto bg-surface/70 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md space-y-6">
      <div className="flex items-center justify-between border-b border-surface-border pb-4">
        <div>
          <span className="text-xs font-mono text-amber-400 uppercase tracking-widest">
            SHAP Interpretability
          </span>
          <h3 className="text-2xl font-bold text-white mt-1">
            Global Feature Importance (Mean |SHAP Value|)
          </h3>
        </div>
        <button
          onClick={() => openExplanation("SHAP")}
          className="flex items-center gap-1.5 text-xs font-mono text-amber-400 hover:underline"
        >
          <HelpCircle className="w-3.5 h-3.5" /> What are SHAP Values?
        </button>
      </div>

      <div className="space-y-5">
        {SHAP_FEATURE_IMPORTANCE.map((item, index) => {
          const percent = (item.meanAbsShap / maxVal) * 100;
          return (
            <div key={item.feature} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div>
                  <span className="font-bold text-white font-mono mr-2">0{index + 1}.</span>
                  <span className="font-semibold text-gray-200">{item.label}</span>
                </div>
                <div className="font-mono text-sm font-bold text-amber-400">
                  {item.meanAbsShap.toFixed(2)}
                </div>
              </div>

              <div className="h-4 w-full bg-surface-border/50 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: `${percent}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 1, delay: index * 0.15, ease: "easeOut" }}
                  className="h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 glow-orange"
                />
              </div>
              <div className="text-xs text-gray-400">{item.desc}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
