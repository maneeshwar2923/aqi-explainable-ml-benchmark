"use client";

import React from "react";
import { motion } from "framer-motion";
import { AQI_COLORS } from "@/theme/colors";

interface AQIGaugeProps {
  value?: number; // 0 to 500
  size?: number;
}

export const AQIGauge: React.FC<AQIGaugeProps> = ({
  value = 245,
  size = 280,
}) => {
  const radius = (size - 40) / 2;
  const circumference = Math.PI * radius; // Semi-circle arc length
  const progress = Math.min(Math.max(value / 500, 0), 1);
  const strokeDashoffset = circumference * (1 - progress);

  const getAQICategory = (val: number) => {
    if (val <= 50) return AQI_COLORS.good;
    if (val <= 100) return AQI_COLORS.satisfactory;
    if (val <= 200) return AQI_COLORS.moderate;
    if (val <= 300) return AQI_COLORS.poor;
    if (val <= 400) return AQI_COLORS.veryPoor;
    return AQI_COLORS.severe;
  };

  const category = getAQICategory(value);

  return (
    <div className="relative flex flex-col items-center justify-center p-6 bg-surface/60 border border-surface-border rounded-3xl backdrop-blur-md">
      <svg width={size} height={size / 1.6} viewBox={`0 0 ${size} ${size / 1.5}`} className="overflow-visible">
        <defs>
          <linearGradient id="aqiGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00E400" />
            <stop offset="20%" stopColor="#55FF00" />
            <stop offset="40%" stopColor="#FFFF00" />
            <stop offset="60%" stopColor="#FF7E00" />
            <stop offset="80%" stopColor="#FF0000" />
            <stop offset="100%" stopColor="#99004C" />
          </linearGradient>
        </defs>

        {/* Background Arc */}
        <path
          d={`M ${20} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 20} ${size / 2}`}
          fill="none"
          stroke="#1f293d"
          strokeWidth="18"
          strokeLinecap="round"
        />

        {/* Value Arc */}
        <motion.path
          d={`M ${20} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 20} ${size / 2}`}
          fill="none"
          stroke="url(#aqiGradient)"
          strokeWidth="18"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          whileInView={{ strokeDashoffset }}
          viewport={{ once: true }}
          transition={{ duration: 1.8, ease: "easeOut" }}
        />
      </svg>

      {/* Center Label */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 translate-y-2 text-center">
        <div className="font-mono text-4xl sm:text-5xl font-extrabold text-white">
          {value}
        </div>
        <div
          className="inline-block px-3 py-0.5 rounded-full text-xs font-semibold mt-1"
          style={{ backgroundColor: category.hex + "20", color: category.hex, borderColor: category.hex + "40", borderWidth: 1 }}
        >
          {category.label.split(" ")[0]}
        </div>
      </div>
    </div>
  );
};
