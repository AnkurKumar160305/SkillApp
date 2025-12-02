import React from 'react';
import { motion } from 'framer-motion';

const RecommendationCard = ({ title, subtitle, details, link, type, index }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass glass-hover rounded-2xl p-6 transition-all duration-300 group cursor-default"
        >
            <div className="flex justify-between items-start mb-3">
                <h3 className="text-lg font-semibold text-white truncate w-3/4 group-hover:text-vampire-blood transition-colors duration-300" title={title}>
                    {title}
                </h3>
                <span className={`text-[10px] font-bold px-2 py-1 rounded-full uppercase tracking-wider ${type === 'Job'
                        ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        : 'bg-green-500/10 text-green-400 border border-green-500/20'
                    }`}>
                    {type}
                </span>
            </div>
            <p className="text-vampire-blood font-medium text-sm mb-2">{subtitle}</p>
            <p className="text-gray-400 text-sm mb-6 line-clamp-2 leading-relaxed">{details}</p>
            {link && (
                <a
                    href={link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-sm font-medium text-white hover:text-vampire-blood transition-colors"
                >
                    View Details
                    <svg className="w-4 h-4 ml-1 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                </a>
            )}
        </motion.div>
    );
};

export default RecommendationCard;
