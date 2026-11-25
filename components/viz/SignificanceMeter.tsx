"use client";

import React from "react";
import { motion } from "framer-motion";
import { Chip } from "@/ui/Chip";
import { ShieldCheck, HelpCircle } from "lucide-react";
import { useStoryStore } from "@/store/useStoryStore";

export const SignificanceMeter: React.FC = () => {
  const { openExplanation } = useStoryStore();

  return (
    <div className="w-full max-w-4xl mx-auto bg-surface/70 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md space-y-8">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-surface-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-amber-400 uppercase tracking-widest">
              Statistical Significance
            </span>
            <Chip status="illustrative" text="Conceptual Explainer" />
          </div>
          <h3 className="text-2xl font-bold text-white mt-1">
            Corrected Resampled Paired t-Test
          </h3>
        </div>
        <button
          onClick={() => openExplanation("t-Test")}
          className="flex items-center gap-1.5 text-xs font-mono text-amber-400 hover:underline"
        >
          <HelpCircle className="w-3.5 h-3.5" /> What is a Paired t-Test?
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        {/* Sweeping Confidence Dial */}
        <div className="relative flex flex-col items-center justify-center p-6 bg-surface/50 border border-surface-border rounded-2xl">
          <svg viewBox="0 0 200 120" className="w-full max-w-xs overflow-visible">
            {/* Background Arc */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#1f293d"
              strokeWidth="16"
              strokeLinecap="round"
            />
            {/* Confidence Arc */}
            <path
              d="M 20 100 A 80 80 0 0 1 150 35"
              fill="none"
              stroke="#eab308"
              strokeWidth="16"
              strokeLinecap="round"
            />
            {/* Sweeping Needle */}
            <motion.line
              x1="100"
              y1="100"
              x2="150"
              y2="35"
              stroke="#ffffff"
              strokeWidth="3"
              strokeLinecap="round"
              initial={{ rotate: -90, originX: "100px", originY: "100px" }}
              whileInView={{ rotate: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: "easeOut" }}
            />
            <circle cx="100" cy="100" r="8" fill="#eab308" />
          </svg>
          <div className="text-center mt-2">
            <div className="font-mono text-2xl font-bold text-amber-400">95% Confidence</div>
            <div className="text-xs text-gray-400 font-mono">Resampled error distribution</div>
          </div>
        </div>

        {/* Model Comparison Graph Node */}
        <div className="space-y-4">
          <div className="text-xs font-mono text-gray-400 uppercase tracking-widest">
            Paired Hypothesis Comparisons
          </div>
          <div className="space-y-3">
            {[
              { vs: "XGBoost", status: "Significant (p < 0.05)", note: "Lower squared error variance" },
              { vs: "LightGBM", status: "Significant (p < 0.05)", note: "Better tree depth optimization" },
              { vs: "Random Forest", status: "Significant (p < 0.05)", note: "Superior categorical handling" },
            ].map((comp, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-surface/80 border border-surface-border/60 flex items-center justify-between"
              >
                <div className="flex items-center gap-2.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span className="font-bold text-white text-sm">CatBoost vs {comp.vs}</span>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono text-emerald-400">{comp.status}</div>
                  <div className="text-[10px] text-gray-400">{comp.note}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
