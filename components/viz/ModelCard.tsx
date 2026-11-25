"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ModelMetric } from "@/data/models";
import { Chip } from "@/ui/Chip";
import { Trophy, Clock, RotateCw, CheckCircle2 } from "lucide-react";

interface ModelCardProps {
  model: ModelMetric;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const [isFlipped, setIsFlipped] = useState(false);
  const [rotateX, setRotateX] = useState(0);
  const [rotateY, setRotateY] = useState(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const card = e.currentTarget;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setRotateX((-y / rect.height) * 15);
    setRotateY((x / rect.width) * 15);
  };

  const handleMouseLeave = () => {
    setRotateX(0);
    setRotateY(0);
  };

  return (
    <div className="perspective-1000 w-full min-h-[360px]">
      <motion.div
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={() => setIsFlipped(!isFlipped)}
        animate={{
          rotateX,
          rotateY: isFlipped ? 180 : rotateY,
        }}
        transition={{ type: "spring", stiffness: 200, damping: 20 }}
        style={{ transformStyle: "preserve-3d" }}
        className={`relative w-full h-full cursor-pointer rounded-3xl p-6 sm:p-8 border transition-all duration-300 ${
          model.isBest
            ? "bg-surface/90 border-amber-500/50 glow-gold"
            : "bg-surface/60 border-surface-border hover:border-gray-600"
        }`}
      >
        {/* Card Front */}
        <div className={`space-y-6 ${isFlipped ? "opacity-0 pointer-events-none" : "opacity-100"}`}>
          <div className="flex items-start justify-between">
            <div>
              <div className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-1">
                {model.category}
              </div>
              <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                {model.name}
                {model.isBest && (
                  <span className="p-1 rounded-full bg-amber-500/20 text-amber-400" title="Best Performer">
                    <Trophy className="w-4 h-4" />
                  </span>
                )}
              </h3>
            </div>
            <Chip status={model.status} />
          </div>

          <p className="text-gray-300 text-sm leading-relaxed">
            {model.description}
          </p>

          {/* Metrics Section */}
          {model.status === "reported" ? (
            <div className="grid grid-cols-3 gap-3 p-4 rounded-2xl bg-surface/80 border border-surface-border/50">
              <div>
                <div className="text-[10px] font-mono uppercase text-gray-400">RMSE</div>
                <div className="font-mono text-xl font-bold text-amber-400">{model.rmse}</div>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase text-gray-400">MAE</div>
                <div className="font-mono text-xl font-bold text-white">{model.mae}</div>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase text-gray-400">R² Score</div>
                <div className="font-mono text-xl font-bold text-emerald-400">{model.r2}</div>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-2xl bg-surface/40 border border-surface-border/50 text-center animate-shimmer space-y-2">
              <div className="text-xs font-mono text-amber-400">Pending Source Data</div>
              <div className="text-xs text-gray-500">Official paper benchmark metrics pending integration</div>
            </div>
          )}

          <div className="flex items-center justify-between text-xs font-mono text-gray-400 pt-2 border-t border-surface-border/40">
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-gray-500" />
              {model.trainTimeSec ? `${model.trainTimeSec}s train time` : "Train time: N/A"}
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <RotateCw className="w-3.5 h-3.5" /> Tap to Flip
            </span>
          </div>
        </div>

        {/* Card Back (Flipped View) */}
        {isFlipped && (
          <div className="absolute inset-0 p-6 sm:p-8 rounded-3xl bg-surface border border-amber-500/40 flex flex-col justify-between">
            <div>
              <div className="text-xs font-mono text-amber-400 uppercase tracking-widest mb-2">
                Algorithm Strengths & Features
              </div>
              <h4 className="text-xl font-bold text-white mb-4">{model.name}</h4>
              <ul className="space-y-2.5">
                {model.strengths.map((st, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs sm:text-sm text-gray-300">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{st}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="text-xs font-mono text-amber-400 flex items-center justify-end gap-1">
              <RotateCw className="w-3.5 h-3.5" /> Tap to Return
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};
