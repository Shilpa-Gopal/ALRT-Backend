
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

interface Citation {
  id: number;
  title: string;
  abstract: string;
  is_relevant: boolean;
  iteration: number;
}

export default function Results() {
  const [citations, setCitations] = useState<Citation[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [keywords, setKeywords] = useState<{include: string[], exclude: string[]}>({
    include: [],
    exclude: []
  });
  const { id } = useParams();

  useEffect(() => {
    const fetchResults = async () => {
      const userId = localStorage.getItem('userId');
      const response = await axios.get(`/api/projects/${id}/citations/filter`, {
        headers: { 'X-User-Id': userId },
        params: { is_relevant: true }
      });
      setCitations(response.data.citations);

      try {
        const metricsResponse = await axios.post(
          `/api/projects/${id}/train`,
          {},
          { headers: { 'X-User-Id': userId } }
        );
        setMetrics(metricsResponse.data.metrics);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };
    fetchResults();
  }, [id]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Results</h1>
      
      {metrics && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Model Performance</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Precision</p>
              <p className="text-lg font-semibold">{(metrics.precision * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Recall</p>
              <p className="text-lg font-semibold">{(metrics.recall * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">F1 Score</p>
              <p className="text-lg font-semibold">{(metrics.f1 * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Accuracy</p>
              <p className="text-lg font-semibold">{(metrics.accuracy * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Keywords</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-medium mb-2">Include Keywords</h3>
            <div className="flex flex-wrap gap-2">
              {keywords.include.map((keyword, index) => (
                <span key={index} className="bg-green-100 px-2 py-1 rounded">
                  {keyword}
                </span>
              ))}
            </div>
          </div>
          <div>
            <h3 className="font-medium mb-2">Exclude Keywords</h3>
            <div className="flex flex-wrap gap-2">
              {keywords.exclude.map((keyword, index) => (
                <span key={index} className="bg-red-100 px-2 py-1 rounded">
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <h2 className="text-xl font-bold p-6 border-b">Relevant Citations</h2>
        <div className="divide-y">
          {citations.map(citation => (
            <div key={citation.id} className="p-6">
              <h3 className="font-semibold mb-2">{citation.title}</h3>
              <p className="text-gray-600 text-sm">{citation.abstract}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-8 flex justify-between">
        <button
          onClick={() => navigate(`/project/${id}/citations`)}
          className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700"
        >
          Back to Citations
        </button>
        <button
          onClick={() => navigate('/home')}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          Finish
        </button>
      </div>
    </div>
  );
}
