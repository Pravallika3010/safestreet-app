import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getRoadDetails } from '../services/api';
import './RoadDetailsPage.css';

const RoadDetailsPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [roadData, setRoadData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    // Check if user is logged in and has proper role
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      
      if (!user || !user.email) {
        console.log('No user found in localStorage, redirecting to login');
        navigate('/login');
        return;
      }
      
      if (user.role !== 'admin' && user.role !== 'authority') {
        console.log('User is not authorized, redirecting to user dashboard');
        navigate('/dashboard');
        return;
      }
      
      // Load road details
      loadRoadDetails();
    } catch (error) {
      console.error('Error checking authentication:', error);
      navigate('/login');
    }
  }, [navigate, id]);

  const loadRoadDetails = async () => {
    if (!id) {
      setError('No road ID provided');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await getRoadDetails(id);
      console.log('Road details:', data);
      setRoadData(data);
    } catch (error) {
      console.error('Error loading road details:', error);
      setError(`Failed to load road details: ${error.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBackClick = () => {
    navigate(-1);
  };

  // Format date string
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (error) {
      return dateString || 'N/A';
    }
  };

  // Safely access nested object properties
  const getNestedValue = (obj, path, defaultValue = 'N/A') => {
    if (!obj) return defaultValue;
    
    const value = path.split('.').reduce((o, key) => (o && o[key] !== undefined) ? o[key] : undefined, obj);
    return value !== undefined ? value : defaultValue;
  };

  return (
    <div className="road-details-page">
      <div className="details-header">
        <button className="back-button" onClick={handleBackClick}>
          ← Back
        </button>
        <h1>Road Details</h1>
      </div>

      {loading ? (
        <div className="loading-spinner">Loading road details...</div>
      ) : error ? (
        <div className="error-message">
          {error}
          <button onClick={loadRoadDetails} className="retry-btn">Retry</button>
        </div>
      ) : roadData ? (
        <div className="road-content">
          <div className="road-image-section">
            {roadData.image_data ? (
              <img 
                src={roadData.image_data} 
                alt="Road" 
                className="road-full-image" 
                onError={(e) => {
                  e.target.src = 'https://via.placeholder.com/800x400?text=Image+Error';
                }}
              />
            ) : (
              <div className="missing-image large">No Image Available</div>
            )}
            <div className={`status-badge large ${roadData.status || 'pending'}`}>
              {roadData.status || 'pending'}
            </div>
          </div>

          <div className="road-info-section">
            <div className="road-info-card">
              <h2>{getNestedValue(roadData, 'name', 'Unnamed Road')}</h2>
              
              <div className="info-group">
                <h3>Location Details</h3>
                <div className="info-row">
                  <span className="info-label">Area:</span>
                  <span className="info-value">{getNestedValue(roadData, 'address.area', 'Not specified')}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Pincode:</span>
                  <span className="info-value">{getNestedValue(roadData, 'address.pincode', 'Not specified')}</span>
                </div>
              </div>
              
              <div className="info-group">
                <h3>Report Information</h3>
                <div className="info-row">
                  <span className="info-label">Reported By:</span>
                  <span className="info-value">{roadData.uploaded_by || 'Unknown'}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Report Date:</span>
                  <span className="info-value">{formatDate(roadData.uploaded_at)}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Status:</span>
                  <span className={`info-value status-text ${roadData.status || 'pending'}`}>
                    {roadData.status || 'pending'}
                  </span>
                </div>
              </div>
              
              {roadData.classified_by && (
                <div className="info-group">
                  <h3>Classification Information</h3>
                  <div className="info-row">
                    <span className="info-label">Classified By:</span>
                    <span className="info-value">{roadData.classified_by}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Classification Date:</span>
                    <span className="info-value">{formatDate(roadData.classified_at)}</span>
                  </div>
                </div>
              )}
              
              {roadData.detailedClassification && roadData.detailedClassification.length > 0 && (
                <div className="info-group">
                  <h3>Road Classification</h3>
                  <div className="classification-results">
                    {roadData.detailedClassification.map((item, index) => (
                      <div key={index} className="result-item">
                        <div className="result-bar" style={{ width: `${item.probability * 100}%` }}></div>
                        <div className="result-label">{item.class}</div>
                        <div className="result-probability">{(item.probability * 100).toFixed(2)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="actions-section">
                <button className="action-btn verify-btn">Verify Report</button>
                <button className="action-btn reject-btn">Reject Report</button>
                <button className="action-btn share-btn">Share Details</button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="no-data-message">
          <p>No road data available or the road ID is invalid.</p>
          <button className="back-button" onClick={handleBackClick}>
            Return to Dashboard
          </button>
        </div>
      )}
    </div>
  );
};

export default RoadDetailsPage; 