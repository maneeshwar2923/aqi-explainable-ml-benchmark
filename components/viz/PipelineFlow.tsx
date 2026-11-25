"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { CHAPTER_CONTENT } from "@/data/content";
import { CheckCircle2, ArrowDown, Database, Filter, Sliders, Calendar, Sparkles } from "lucide-react";

export const PipelineFlow: React.FC = () => {
  const [activeStep, setActiveStep] = useState<number>(1);
  const steps = CHAPTER_CONTENT.preprocessing.steps;

  const icons = [Filter, Database, Sliders, Calendar, Sparkles];

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      {/* Pipeline Navigation / Selector */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {steps.map((s, idx) => {
          const Icon = icons[idx];
          const isSelected = activeStep === s.step;
          return (
            <button
              key={s.step}
              onClick={() => setActiveStep(s.step)}
              className={`p-4 rounded-2xl border text-left transition-all duration-300 ${
                isSelected
                  ? "bg-amber-500/10 border-amber-500/50 glow-orange"
                  : "bg-surface/50 border-surface-border hover:border-gray-600"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-xs text-amber-400 font-bold">
                  STEP 0{s.step}
                </span>
                <Icon className={`w-4 h-4 ${isSelected ? "text-amber-400" : "text-gray-500"}`} />
              </div>
              <div className="font-bold text-white text-sm line-clamp-1">{s.title}</div>
            </button>
          );
        })}
      </div>

      {/* Active Pipeline Card Detail */}
      <motion.div
        key={activeStep}
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="bg-surface/80 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md shadow-2xl"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold font-mono">
            0{activeStep}
          </div>
          <div>
            <h3 className="text-2xl font-bold text-white">
              {steps[activeStep - 1].title}
            </h3>
            <p className="text-gray-400 text-sm">
              {steps[activeStep - 1].desc}
            </p>
          </div>
        </div>

        {/* Step-Specific Special Visualizations */}
        {activeStep === 4 && (
          <div className="mt-6 p-6 rounded-2xl bg-surface/90 border border-amber-500/20">
            <div className="text-xs font-mono text-amber-400 uppercase tracking-widest mb-4">
              Temporal Splitting Mechanism (2015 - 2020)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
                <div className="text-xs font-mono text-emerald-400 font-bold">TRAIN SET (2015 - 2018)</div>
                <div className="font-mono text-3xl font-extrabold text-white my-1">13,214</div>
                <div className="text-xs text-gray-400">Strict baseline observation records</div>
              </div>
              <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/30">
                <div className="text-xs font-mono text-blue-400 font-bold">TEST SET (2019 - 2020)</div>
                <div className="font-mono text-3xl font-extrabold text-white my-1">11,410</div>
                <div className="text-xs text-gray-400">Future out-of-sample evaluation</div>
              </div>
            </div>
          </div>
        )}

        {activeStep === 5 && (
          <div className="mt-6 p-6 rounded-2xl bg-surface/90 border border-amber-500/20">
            <div className="text-xs font-mono text-amber-400 uppercase tracking-widest mb-4">
              Z-Score Normalization (Train Fit Only)
            </div>
            <div className="h-32 w-full flex items-center justify-center">
              <svg viewBox="0 0 300 80" className="w-full h-full max-w-md overflow-visible">
                {/* Lumpy Raw Bell Curve */}
                <path
                  d="M 10 70 Q 60 10 120 40 T 200 65 T 290 70"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                />
                {/* Standardized Gaussian Curve */}
                <path
                  d="M 10 70 Q 150 -10 290 70"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="3"
                />
              </svg>
            </div>
            <div className="flex justify-between text-xs font-mono text-gray-400 mt-2 px-4">
              <span className="text-red-400">--- Raw Lumpy Pollutant Distribution</span>
              <span className="text-emerald-400">── Z-Score Standardized Gaussian</span>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};
