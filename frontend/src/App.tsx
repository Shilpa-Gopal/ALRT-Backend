
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/auth/Login';
import Signup from './components/auth/Signup';
import Home from './components/Home';
import NewProject from './components/project/NewProject';
import KeywordSelection from './components/project/KeywordSelection';
import CitationLabeling from './components/project/CitationLabeling';
import Results from './components/project/Results';

function PrivateRoute({ children }: { children: React.ReactElement }) {
  const userId = localStorage.getItem('userId');
  return userId ? children : <Navigate to="/login" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/home" element={<PrivateRoute><Home /></PrivateRoute>} />
          <Route path="/project/new" element={<PrivateRoute><NewProject /></PrivateRoute>} />
          <Route path="/project/:id/keywords" element={<PrivateRoute><KeywordSelection /></PrivateRoute>} />
          <Route path="/project/:id/citations" element={<PrivateRoute><CitationLabeling /></PrivateRoute>} />
          <Route path="/project/:id/results" element={<PrivateRoute><Results /></PrivateRoute>} />
          <Route path="/" element={<Navigate to="/login" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
