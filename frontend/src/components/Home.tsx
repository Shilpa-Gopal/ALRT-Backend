
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

interface Project {
  id: number;
  name: string;
  created_at: string;
  current_iteration: number;
}

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);

  useEffect(() => {
    const fetchProjects = async () => {
      const userId = localStorage.getItem('userId');
      const response = await axios.get('/api/projects', {
        headers: { 'X-User-Id': userId }
      });
      setProjects(response.data.projects);
    };
    fetchProjects();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold">My Projects</h1>
        <Link
          to="/project/new"
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          New Project
        </Link>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((project) => (
          <Link
            key={project.id}
            to={`/project/${project.id}/citations`}
            className="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow"
          >
            <h3 className="text-lg font-semibold mb-2">{project.name}</h3>
            <p className="text-gray-600">Iteration: {project.current_iteration}</p>
            <p className="text-gray-600 text-sm">
              Created: {new Date(project.created_at).toLocaleDateString()}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
