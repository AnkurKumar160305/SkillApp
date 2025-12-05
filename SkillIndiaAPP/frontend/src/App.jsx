import React, { useState } from 'react';
import { motion } from 'framer-motion';
import SkillInput from './components/SkillInput';
import RecommendationCard from './components/RecommendationCard';

import Background3D from './components/Background3D';

function App() {
  const [skills, setSkills] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async () => {
    if (skills.length === 0) return;

    setLoading(true);
    setHasSearched(true);
    const skillsStr = skills.join(' ');

    try {
      // Fetch Jobs
      const jobsRes = await fetch('http://localhost:8000/recommend_jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skills: skillsStr }),
      });
      const jobsData = await jobsRes.json();
      setJobs(jobsData.recommendations || []);

      // Fetch Courses
      const coursesRes = await fetch('http://localhost:8000/recommend_courses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skills: skillsStr }),
      });
      const coursesData = await coursesRes.json();
      setCourses(coursesData.recommendations || []);

    } catch (error) {
      console.error("Error fetching recommendations:", error);
      alert("Failed to fetch recommendations. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-gray-300 font-sans selection:bg-vampire-red/30 selection:text-white overflow-hidden relative">
      <Background3D />

      <div className="container mx-auto px-4 py-16 relative z-10">
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          <h1 className="text-5xl md:text-7xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-500 mb-6 tracking-tight">
            Skill<span className="text-vampire-red">Forge</span>
          </h1>
          <p className="text-lg text-gray-500 max-w-xl mx-auto font-light">
            Discover your next career move with AI-powered recommendations tailored to your unique skillset.
          </p>
        </motion.header>

        <SkillInput skills={skills} setSkills={setSkills} />

        <div className="text-center mb-20">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSearch}
            disabled={loading || skills.length === 0}
            className="bg-white text-black hover:bg-gray-200 font-semibold py-4 px-10 rounded-full shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-lg tracking-wide"
          >
            {loading ? 'Analyzing...' : 'Reveal Path'}
          </motion.button>
        </div>

        {(jobs.length > 0 || courses.length > 0) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-12"
          >
            <section>
              <div className="flex items-center mb-8">
                <div className="w-1 h-8 bg-vampire-red mr-4 rounded-full"></div>
                <h2 className="text-2xl font-bold text-white tracking-wide">
                  Career Opportunities
                </h2>
              </div>
              <div className="space-y-4">
                {jobs.map((job, index) => (
                  <RecommendationCard
                    key={index}
                    index={index}
                    title={job.job_post}
                    subtitle={job.company}
                    details={`${job.job_location} • ${job.required_skills}`}
                    type="Job"
                  />
                ))}
              </div>
            </section>

            <section>
              <div className="flex items-center mb-8">
                <div className="w-1 h-8 bg-blue-500 mr-4 rounded-full"></div>
                <h2 className="text-2xl font-bold text-white tracking-wide">
                  Learning Paths
                </h2>
              </div>
              <div className="space-y-4">
                {courses.map((course, index) => (
                  <RecommendationCard
                    key={index}
                    index={index}
                    title={course.course_title}
                    subtitle={`${course.level} • ${course.price === 'Free' || course.price === 'TRUE' ? 'Paid' : course.price}`}
                    details={course.url}
                    link={course.url}
                    type="Course"
                  />
                ))}
              </div>
            </section>
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default App;
