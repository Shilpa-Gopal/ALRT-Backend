
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function NewProject() {
  const [name, setName] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const userId = localStorage.getItem('userId');
      const response = await axios.post('/api/projects', 
        { name },
        { headers: { 'X-User-Id': userId } }
      );
      navigate(`/project/${response.data.project.id}/keywords`);
    } catch (error) {
      alert('Failed to create project');
    }
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Create New Project</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Project Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full border rounded-md shadow-sm p-2"
            required
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
        >
          Create Project
        </button>
      </form>
    </div>
  );
}
