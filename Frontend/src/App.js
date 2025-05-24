import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import LandingPage from './pages/LandingPage';
import Login from './pages/Login';
import UserDashboard from './pages/UserDashboard';
import AdminDashboard from './pages/AdminDashboard';

// Protected route component
const ProtectedRoute = ({ element, requiredRole }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is authenticated
    try {
      const userData = JSON.parse(localStorage.getItem('user') || '{}');
      if (userData && userData.email) {
        console.log('User authenticated:', userData.email);
        setUserRole(userData.role || 'user');
        setIsAuthenticated(true);
      } else {
        console.log('No user data found in localStorage');
        setIsAuthenticated(false);
      }
    } catch (error) {
      console.error('Error parsing user data:', error);
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  }, []);

  if (loading) {
    return <div className="loading-screen">Loading...</div>;
  }

  if (!isAuthenticated) {
    console.log('Not authenticated, redirecting to login');
    return <Navigate to="/login" />;
  }

  // Check if user has the required role
  if (requiredRole && userRole !== requiredRole) {
    console.log(`User role ${userRole} doesn't match required role ${requiredRole}`);
    return userRole === 'admin' ? 
      <Navigate to="/admin" /> : 
      <Navigate to="/dashboard" />;
  }

  console.log('Rendering protected route with role:', userRole);
  return element;
};

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<ProtectedRoute element={<UserDashboard />} />} />
          <Route path="/admin" element={<ProtectedRoute element={<AdminDashboard />} requiredRole="admin" />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
