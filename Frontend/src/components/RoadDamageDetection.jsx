import React, { useState } from 'react';
import { detectRoadDamage } from '../services/api';
import axios from 'axios';
import { testLogin, adminClassifyImage } from '../services/api';

const RoadDamageDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [testMode, setTestMode] = useState('test-login');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError('');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    console.log('Starting road damage detection');
    console.log('Selected file:', selectedFile.name, 'Size:', selectedFile.size);
    
    setLoading(true);
    setError('');

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.readAsDataURL(selectedFile);
      
      reader.onload = async () => {
        try {
          console.log('Image converted to base64, sending to backend');
          // Send image to backend for road damage detection
          const result = await detectRoadDamage(reader.result);
          
          console.log('Detection results received:', result);
          setResults(result);
          
          if (!result.is_road) {
            setError('This image is not a road. Please upload a road image.');
          }
        } catch (err) {
          console.error('Error during detection:', err);
          setError(err.message || 'Error processing image');
        } finally {
          setLoading(false);
        }
      };
      
      reader.onerror = (error) => {
        console.error('Error reading file:', error);
        setError('Error reading image file');
        setLoading(false);
      };
    } catch (err) {
      console.error('Exception in upload handler:', err);
      setError(err.message || 'Error processing image');
      setLoading(false);
    }
  };

  const handleTest = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      if (testMode === 'test-login') {
        // Test the login endpoint
        const loginResult = await testLogin();
        setResult(loginResult);
      } else if (testMode === 'direct-classify') {
        // Direct test of the admin/classify endpoint
        const token = localStorage.getItem('token');
        
        if (!token) {
          // Try to login first
          await testLogin();
        }
        
        // Now try a direct call with axios
        const classifyResult = await axios.post('/api/admin/classify', 
          { image_id: 'test123', mode: 'road_damage' },
          { 
            headers: { 
              'Authorization': `Bearer ${localStorage.getItem('token')}`,
              'Content-Type': 'application/json'
            } 
          }
        );
        
        setResult(classifyResult.data);
      } else if (testMode === 'api-classify') {
        // Use the API service method
        const classifyResult = await adminClassifyImage('test123', 'road_damage');
        setResult(classifyResult);
      }
    } catch (err) {
      console.error('Test failed:', err);
      setError(err.toString());
      
      // Add more detailed error info
      if (err.response) {
        setError(`${err.toString()} - Status: ${err.response.status}, Data: ${JSON.stringify(err.response.data)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const renderDamageDetectionResults = () => {
    if (!results) return null;
    
    // Extract classification results
    const classificationResults = results.classification || {};
    const detectionResults = results.detection || {};
    
    // Determine if road is damaged
    const isDamaged = results.is_damaged || 
                     (classificationResults.class_name === 'Damaged') ||
                     (results.detections && results.detections.length > 0);
    
    // Get confidence level
    const damageConfidence = results.damage_confidence || 
                           classificationResults.confidence || 
                           0;
    
    // Get detections array from different possible formats
    const detections = results.detections || 
                      (detectionResults.detections || []);
    
    return (
      <div className="detection-results">
        <h3>Road Analysis Results:</h3>
        
        <div className="result-item">
          <strong>Road Type:</strong> {results.road_type || 'Asphalt'}
        </div>
        
        <div className="result-item">
          <strong>Damage Status:</strong> {isDamaged ? 'Damaged' : 'Normal'}
          {damageConfidence > 0 && (
            <span className="confidence"> (Confidence: {(damageConfidence * 100).toFixed(2)}%)</span>
          )}
        </div>
        
        {results.summary && (
          <div className="result-item">
            <strong>Summary:</strong> {results.summary}
          </div>
        )}
        
        {results.processing_time && (
          <div className="result-item">
            <strong>Processing Time:</strong> 
            {results.processing_time.detection && 
              `Detection: ${results.processing_time.detection.toFixed(2)}s`} 
            {results.processing_time.classification && 
              `, Classification: ${results.processing_time.classification.toFixed(2)}s`}
          </div>
        )}
        
        {detections && detections.length > 0 ? (
          <div className="damages-list">
            <h4>Detected Damages:</h4>
            {detections.map((damage, index) => (
              <div key={index} className="damage-item">
                <div><strong>Type:</strong> {damage.label || damage.class_name || 'Unknown'}</div>
                <div><strong>Severity:</strong> {damage.severity || 'Medium'}</div>
                <div><strong>Confidence:</strong> {((damage.confidence || 0) * 100).toFixed(2)}%</div>
                {damage.bbox && (
                  <div><strong>Location:</strong> [x1: {Math.round(damage.bbox[0])}, y1: {Math.round(damage.bbox[1])}, 
                  x2: {Math.round(damage.bbox[2])}, y2: {Math.round(damage.bbox[3])}]</div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p>{isDamaged ? 'General road damage detected without specific locations.' : 'No damages detected in this road.'}</p>
        )}
      </div>
    );
  };

  return (
    <div className="road-damage-detection-container">
      <h2>Road Damage Detection</h2>
      <p>Upload a road image to analyze for damages using our advanced AI model.</p>
      
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="file-input"
        />
        
        {previewUrl && (
          <div className="preview-container">
            <img src={previewUrl} alt="Preview" className="preview-image" />
          </div>
        )}
      </div>

      <button 
        onClick={handleAnalyze} 
        disabled={!selectedFile || loading}
        className="analyze-button"
      >
        {loading ? 'Analyzing...' : 'Analyze Road Damage'}
      </button>

      {renderDamageDetectionResults()}

      {error && <div className="error-message">{error}</div>}
      
      <div className="debug-panel" style={{ margin: '20px', padding: '20px', border: '1px solid #ccc' }}>
        <h3>API Debug Panel</h3>
        
        <div style={{ marginBottom: '20px' }}>
          <label>
            Test Mode:
            <select 
              value={testMode} 
              onChange={(e) => setTestMode(e.target.value)}
              style={{ marginLeft: '10px' }}
            >
              <option value="test-login">Test Login</option>
              <option value="direct-classify">Direct Admin Classify</option>
              <option value="api-classify">API Admin Classify</option>
            </select>
          </label>
        </div>
        
        <button 
          onClick={handleTest}
          disabled={loading}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: '#3498db', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Testing...' : 'Run Test'}
        </button>
        
        {result && (
          <div className="result" style={{ margin: '20px 0' }}>
            <h4>Test Result:</h4>
            <pre style={{ backgroundColor: '#f8f9fa', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
      
      <style jsx>{`
        .road-damage-detection-container {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        
        .upload-section {
          margin: 20px 0;
        }
        
        .preview-container {
          margin-top: 15px;
        }
        
        .preview-image {
          max-width: 100%;
          max-height: 400px;
          border-radius: 8px;
        }
        
        .analyze-button {
          padding: 10px 20px;
          background-color: #4caf50;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 16px;
          margin-bottom: 20px;
        }
        
        .analyze-button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .detection-results {
          margin-top: 30px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #f9f9f9;
        }
        
        .result-item {
          margin-bottom: 10px;
        }
        
        .damages-list {
          margin-top: 20px;
        }
        
        .damage-item {
          margin-bottom: 15px;
          padding: 10px;
          background-color: #fff;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .error-message {
          color: #d32f2f;
          margin-top: 15px;
          padding: 10px;
          background-color: #ffebee;
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

export default RoadDamageDetection; 