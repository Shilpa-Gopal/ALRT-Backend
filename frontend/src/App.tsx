
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/auth/Login';
import Signup from './components/auth/Signup';
import Home from './components/Home';
import NewProject from './components/project/NewProject';
import KeywordSelection from './components/project/KeywordSelection';
import CitationLabeling from './components/project/CitationLabeling';
import Results from './components/project/Results';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/home" element={<Home />} />
          <Route path="/project/new" element={<NewProject />} />
          <Route path="/project/:id/keywords" element={<KeywordSelection />} />
          <Route path="/project/:id/citations" element={<CitationLabeling />} />
          <Route path="/project/:id/results" element={<Results />} />
          <Route path="/" element={<Navigate to="/login" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
