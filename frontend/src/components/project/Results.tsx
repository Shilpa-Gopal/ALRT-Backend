
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
    </div>
  );
}
