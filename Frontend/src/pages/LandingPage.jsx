// LandingPage.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();
  
  return (
    <div className="landing-page">
      <div className="landing-image-section">
        {/* Background image with content overlay */}
        <div className="overlay-content">
          <h1 className="landing-title">Welcome to SAFESTREET</h1>
          <p className="landing-subtitle">
            AI-powered road condition monitoring system for safer communities.
          </p>
          
          <div className="info-section">
            <div className="info-item">
              <h3>Smart Detection</h3>
              <p>Our AI-powered system detects 94% of road hazards, including potholes, cracks, and structural damage with precision that exceeds manual inspection.</p>
            </div>
            
            <div className="info-item">
              <h3>Real-time Analytics</h3>
              <p>Analyze road conditions instantly with our advanced neural networks that process data 200x faster than traditional methods, prioritizing repairs where they matter most.</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="top-corner-button">
        <button 
          className="btn-login"
          onClick={() => navigate('/login')}
        >
          Login
        </button>
      </div>
    </div>
  );
};

export default LandingPage;