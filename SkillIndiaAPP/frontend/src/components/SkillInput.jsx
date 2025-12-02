import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const SkillInput = ({ skills, setSkills }) => {
    const [input, setInput] = useState('');

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && input.trim()) {
            e.preventDefault();
            if (!skills.includes(input.trim())) {
                setSkills([...skills, input.trim()]);
            }
            setInput('');
        }
    };

    const removeSkill = (skillToRemove) => {
        setSkills(skills.filter(skill => skill !== skillToRemove));
    };

    return (
        <div className="w-full max-w-2xl mx-auto mb-12">
            <label className="block text-gray-400 text-sm font-medium mb-3 tracking-wide">
                ADD YOUR SKILLS
            </label>
            <div className="glass rounded-2xl p-3 flex flex-wrap items-center min-h-[60px] transition-all duration-300 focus-within:ring-1 focus-within:ring-vampire-red/50">
                <AnimatePresence>
                    {skills.map((skill, index) => (
                        <motion.span
                            key={skill}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            className="bg-vampire-red/20 text-vampire-blood border border-vampire-red/30 px-4 py-1.5 rounded-full text-sm font-medium mr-2 mb-2 flex items-center backdrop-blur-md"
                        >
                            {skill}
                            <button
                                onClick={() => removeSkill(skill)}
                                className="ml-2 text-vampire-blood/70 hover:text-vampire-blood transition-colors"
                            >
                                &times;
                            </button>
                        </motion.span>
                    ))}
                </AnimatePresence>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="bg-transparent outline-none text-white flex-grow min-w-[150px] placeholder-gray-600 px-2 py-1"
                    placeholder="Type skill & press Enter..."
                />
            </div>
        </div>
    );
};

export default SkillInput;
