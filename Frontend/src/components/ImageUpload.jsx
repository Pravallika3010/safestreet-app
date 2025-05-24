import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { saveImageToDatabase } from '../services/api';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [classification, setClassification] = useState('');
  const [detailedClassification, setDetailedClassification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setClassification('');
      setDetailedClassification(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    console.log('Starting image upload and classification process');
    console.log('Selected file:', selectedFile.name, 'Size:', selectedFile.size);
    
    setLoading(true);
    setError('');

    try {
      // Convert image to base64
      console.log('Converting image to base64');
      const reader = new FileReader();
      reader.readAsDataURL(selectedFile);
      
      reader.onload = async () => {
        try {
          console.log('Image converted to base64, sending to backend');
          // Send image to backend for binary classification (road/not road)
          console.log('Calling classifyImage with mode: binary');
          const result = await classifyImage(reader.result, 'binary');
          
          console.log('Classification result received:', result);
          
          // Set classification results
          const isRoad = result.is_road;
          console.log('Is road:', isRoad);
          
          setClassification(isRoad ? 'road' : 'not_road');
          setDetailedClassification(result.detailed_classification);
          
          // Show appropriate message based on classification result
          if (!isRoad) {
            console.log('Not a road image, showing error message');
            setError('This image is not a road. Only road images can be uploaded for verification.');
          } else {
            console.log('Road image detected, clearing error messages');
            setError('');
          }
        } catch (err) {
          console.error('Error during classification:', err);
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

  // Function to save the image to database
  const handleSaveToDatabase = async () => {
    if (!selectedFile || classification !== 'road') {
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Convert image to base64 again
      const reader = new FileReader();
      reader.readAsDataURL(selectedFile);
      
      reader.onload = async () => {
        try {
          // Save the image to database
          const result = await saveImageToDatabase(reader.result);
          
          if (result.success) {
            // Show success message
            setError('');
            alert('Image saved to database successfully! Admin will verify it.');
            
            // Reset form
            setSelectedFile(null);
            setPreviewUrl('');
            setClassification('');
            setDetailedClassification(null);
          } else {
            setError('Failed to save image to database.');
          }
        } catch (err) {
          setError(err.message || 'Error saving image to database');
        } finally {
          setLoading(false);
        }
      };
      
      reader.onerror = () => {
        setError('Error reading image file');
        setLoading(false);
      };
    } catch (err) {
      setError(err.message || 'Error saving image to database');
      setLoading(false);
    }
  };

  return (
    <div className="image-upload-container">
      <h2>Upload and Classify Image</h2>
      
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
        onClick={handleUpload} 
        disabled={!selectedFile || loading}
        className="upload-button"
      >
        {loading ? 'Processing...' : 'Upload and Classify'}
      </button>

      {classification && (
        <div className="classification-result">
          <h3>Classification Result:</h3>
          <p>This image is classified as: <strong>{classification}</strong></p>
          
          {detailedClassification && (
            <div className="detailed-classification">
              <h4>Detailed Classification:</h4>
              <ul>
                {detailedClassification.map((result, index) => (
                  <li key={index}>
                    {result.class}: {(result.probability * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Only show save button if it's a road image */}
          {classification === 'road' && (
            <button 
              onClick={handleSaveToDatabase} 
              disabled={loading}
              className="save-button"
            >
              {loading ? 'Saving...' : 'Save to Database for Admin Verification'}
            </button>
          )}
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default ImageUpload; 