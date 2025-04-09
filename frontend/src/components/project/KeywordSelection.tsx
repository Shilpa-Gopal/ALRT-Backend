
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function KeywordSelection() {
  const [includeKeywords, setIncludeKeywords] = useState<string[]>([]);
  const [excludeKeywords, setExcludeKeywords] = useState<string[]>([]);
  const [suggestedKeywords, setSuggestedKeywords] = useState<Array<{word: string, score: number}>>([]);
  const [newKeyword, setNewKeyword] = useState('');
  const { id } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchKeywords = async () => {
      const userId = localStorage.getItem('userId');
      const response = await axios.get(`/api/projects/${id}/keywords`, {
        headers: { 'X-User-Id': userId }
      });
      setIncludeKeywords(response.data.selected_keywords?.include || []);
      setExcludeKeywords(response.data.selected_keywords?.exclude || []);
      setSuggestedKeywords(response.data.suggested_keywords || []);
    };
    fetchKeywords();
  }, [id]);

  const handleAddKeyword = (type: 'include' | 'exclude') => {
    if (!newKeyword.trim()) return;
    if (type === 'include') {
      setIncludeKeywords([...includeKeywords, newKeyword.trim()]);
    } else {
      setExcludeKeywords([...excludeKeywords, newKeyword.trim()]);
    }
    setNewKeyword('');
  };

  const handleSave = async () => {
    const userId = localStorage.getItem('userId');
    await axios.put(
      `/api/projects/${id}/keywords`,
      { include: includeKeywords, exclude: excludeKeywords },
      { headers: { 'X-User-Id': userId } }
    );
    navigate(`/project/${id}/citations`);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Select Keywords</h1>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-4">Suggested Keywords</h3>
          <div className="flex flex-wrap gap-2 mb-6">
            {suggestedKeywords.map((keyword, index) => (
              <button
                key={index}
                onClick={() => handleAddKeyword('include', keyword.word)}
                className="bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-full text-sm"
                title={`Relevance score: ${keyword.score.toFixed(3)}`}
              >
                {keyword.word}
              </button>
            ))}
          </div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Add Custom Keywords
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={newKeyword}
              onChange={(e) => setNewKeyword(e.target.value)}
              className="flex-1 border rounded-md p-2"
              placeholder="Enter a keyword"
            />
            <button
              onClick={() => handleAddKeyword('include')}
              className="bg-green-600 text-white px-4 py-2 rounded-md"
            >
              Include
            </button>
            <button
              onClick={() => handleAddKeyword('exclude')}
              className="bg-red-600 text-white px-4 py-2 rounded-md"
            >
              Exclude
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium mb-2">Include Keywords</h3>
            <div className="space-y-2">
              {includeKeywords.map((keyword, index) => (
                <div key={index} className="flex items-center gap-2 bg-green-100 p-2 rounded">
                  <span>{keyword}</span>
                  <button
                    onClick={() => setIncludeKeywords(includeKeywords.filter((_, i) => i !== index))}
                    className="text-red-600"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-medium mb-2">Exclude Keywords</h3>
            <div className="space-y-2">
              {excludeKeywords.map((keyword, index) => (
                <div key={index} className="flex items-center gap-2 bg-red-100 p-2 rounded">
                  <span>{keyword}</span>
                  <button
                    onClick={() => setExcludeKeywords(excludeKeywords.filter((_, i) => i !== index))}
                    className="text-red-600"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={handleSave}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
        >
          Continue to Citations
        </button>
      </div>
    </div>
  );
}
