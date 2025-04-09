
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

interface Citation {
  id: number;
  title: string;
  abstract: string;
  is_relevant?: boolean;
}

export default function CitationLabeling() {
  const [citations, setCitations] = useState<Citation[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const { id } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCitations = async () => {
      const userId = localStorage.getItem('userId');
      const response = await axios.get(`/api/projects/${id}/citations/filter`, {
        headers: { 'X-User-Id': userId }
      });
      setCitations(response.data.citations);
    };
    fetchCitations();
  }, [id]);

  const handleLabel = async (isRelevant: boolean) => {
    if (!citations[currentIndex]) return;

    const userId = localStorage.getItem('userId');
    await axios.put(
      `/api/projects/${id}/citations/${citations[currentIndex].id}`,
      { is_relevant: isRelevant },
      { headers: { 'X-User-Id': userId } }
    );

    if (currentIndex < citations.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      navigate(`/project/${id}/results`);
    }
  };

  if (!citations.length) {
    return <div className="text-center py-8">Loading citations...</div>;
  }

  const currentCitation = citations[currentIndex];

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-4">
          <span className="text-sm text-gray-500">
            Citation {currentIndex + 1} of {citations.length}
          </span>
        </div>
        
        <h2 className="text-xl font-bold mb-4">{currentCitation.title}</h2>
        <p className="text-gray-700 mb-6 whitespace-pre-wrap">{currentCitation.abstract}</p>

        <div className="flex justify-center gap-4">
          <button
            onClick={() => handleLabel(false)}
            className="px-6 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Not Relevant
          </button>
          <button
            onClick={() => handleLabel(true)}
            className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
          >
            Relevant
          </button>
        </div>
      </div>
    </div>
  );
}
