import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAdminImages, adminClassifyImage, updateImageAddress, classifyImage } from '../services/api';
import './AdminDashboard.css';

const AdminDashboard = () => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedImage, setSelectedImage] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [classifying, setClassifying] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [editingAddress, setEditingAddress] = useState(false);
  const [addressData, setAddressData] = useState({
    name: '',
    area: '',
    pincode: ''
  });
  const [savingAddress, setSavingAddress] = useState(false);
  const [visualizedImage, setVisualizedImage] = useState(null);
  const canvasRef = useRef(null);
  const navigate = useNavigate();

  const loadImages = useCallback(async () => {
    setLoading(true);
    setError('');
    
    try {
      console.log('Loading admin images, page:', currentPage);
      const response = await getAdminImages(currentPage);
      console.log('Admin images response:', response);
      
      // Verify we have images data
      if (!response || !response.images) {
        throw new Error('No images data returned from API');
      }
      
      setImages(response.images || []);
      setTotalPages(response.pages || 1);
      
      if (response.images.length === 0) {
        setError('No images found in the database.');
      }
    } catch (error) {
      console.error('Error loading images:', error);
      setError(`Failed to load images: ${error.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, [currentPage]);

  useEffect(() => {
    // Check if user is logged in and is admin
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      
      if (!user || !user.email) {
        console.log('No user found in localStorage, redirecting to login');
        navigate('/login');
        return;
      }
      
      if (user.role !== 'admin') {
        console.log('User is not an admin, redirecting to user dashboard');
        navigate('/dashboard');
        return;
      }
      
      console.log('Admin authenticated:', user.email);
      
      // Load admin images
      loadImages();
    } catch (error) {
      console.error('Error checking authentication:', error);
      navigate('/login');
    }
  }, [navigate, loadImages]);

  // Function to safely access nested object properties
  const getNestedValue = (obj, path, defaultValue = 'N/A') => {
    if (!obj) return defaultValue;
    
    const value = path.split('.').reduce((o, key) => (o && o[key] !== undefined) ? o[key] : undefined, obj);
    return value !== undefined ? value : defaultValue;
  };

  const handleImageSelect = (image) => {
    console.log('Selected image:', image);
    setSelectedImage(image);
    setShowDetails(false);
    setEditingAddress(false);
    
    // Initialize address data from image if available
    setAddressData({
      name: getNestedValue(image, 'name', ''),
      area: getNestedValue(image, 'address.area', ''),
      pincode: getNestedValue(image, 'address.pincode', '')
    });
    
    // If the image is already classified, show the classification results
    if (image.status === 'classified') {
      console.log('Image is already classified, showing results');
      
      // Set classification result
      setClassificationResult({
        success: true,
        classification: {
          is_road: image.is_road,
          damage_type: image.damage_type,
          severity: image.severity
        },
        detections: image.detections || []
      });
      
      // If there's an output image URL, display it
      if (image.output_image_url) {
        const fullImageUrl = `http://localhost:5001${image.output_image_url}`;
        console.log('Setting visualized image from stored output_image_url:', fullImageUrl);
        setVisualizedImage(fullImageUrl);
      } else {
        // If no output image but we have detections, we could draw them
        setVisualizedImage(`http://localhost:5001/uploads/${image.filename}`);
      }
    } else {
      // For unclassified images, clear any previous classification data
      setClassificationResult(null);
      setVisualizedImage(null);
    }
    
    // Don't automatically visualize detections when selecting an image
    // We'll only show detections after the user clicks the classify button
  };

  const drawBoundingBoxes = (image) => {
    if (!image || !image.detections || !image.detections.length) {
      console.warn('Cannot draw bounding boxes: missing image data or detections');
      return;
    }
    
    // Create a new image element to load the original image
    const img = new Image();
    
    img.onload = () => {
      // Create a canvas to draw on
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Set canvas dimensions to match the image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw the original image on the canvas
      ctx.drawImage(img, 0, 0);
      
      // Draw each bounding box
      image.detections.forEach(detection => {
        // Get bounding box coordinates
        let x, y, width, height;
        
        // Handle different bbox formats
        if (Array.isArray(detection.bbox) && detection.bbox.length === 4) {
          // Check if it's [x, y, width, height] format or [x1, y1, x2, y2] format
          if (detection.bbox[2] > detection.bbox[0] && detection.bbox[3] > detection.bbox[1]) {
            // It's [x1, y1, x2, y2] format
            x = detection.bbox[0];
            y = detection.bbox[1];
            width = detection.bbox[2] - detection.bbox[0];
            height = detection.bbox[3] - detection.bbox[1];
          } else {
            // It's [x, y, width, height] format
            [x, y, width, height] = detection.bbox;
          }
        } else if (detection.bbox && typeof detection.bbox === 'object') {
          // Extract coordinates from object format
          const { x1, y1, x2, y2 } = detection.bbox;
          x = x1;
          y = y1;
          width = x2 - x1;
          height = y2 - y1;
        } else {
          console.error('Invalid bounding box format:', detection.bbox);
          return;
        }
        
        // Get damage information
        const damageType = detection.class || detection.class_name || detection.label || 'Damage';
        const confidence = detection.confidence || 0;
        const severity = detection.severity || 'unknown';
        
        // Set box style based on damage type and severity
        let boxColor;
        if (damageType.toLowerCase().includes('pothole')) {
          boxColor = '#FF6B00'; // Orange for potholes
        } else if (damageType.toLowerCase().includes('crack')) {
          boxColor = '#FFCC00'; // Yellow for cracks
        } else {
          boxColor = '#FF0000'; // Default red for unknown
        }
        
        // Draw the bounding box
        ctx.lineWidth = 3;
        ctx.strokeStyle = boxColor;
        ctx.fillStyle = boxColor + '33'; // Add 20% opacity
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.stroke();
        ctx.fill();
        
        // Draw label background
        const label = `${damageType} (${Math.round(confidence * 100)}%) - ${severity}`;
        ctx.font = '14px Arial';
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = boxColor;
        ctx.fillRect(x, y - 20, textWidth + 10, 20);
        
        // Draw label text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x + 5, y - 5);
      });
      
      // Convert canvas to data URL
      const visualizedDataUrl = canvas.toDataURL('image/jpeg');
      setVisualizedImage(visualizedDataUrl);
    };
    
    // Set source of image - if data_url is available, use it, otherwise use image_data
    if (image.data_url) {
      img.src = image.data_url;
    } else if (image.image_data) {
      // Check if image_data already has data: prefix
      if (image.image_data.startsWith('data:')) {
        img.src = image.image_data;
      } else {
        // Assume it's a base64 string without prefix
        img.src = `data:image/jpeg;base64,${image.image_data}`;
      }
    }
  };

  const handleAddressChange = (e) => {
    const { name, value } = e.target;
    
    if (name === 'name') {
      setAddressData(prev => ({ ...prev, name: value }));
    } else if (name === 'area') {
      setAddressData(prev => ({ ...prev, area: value }));
    } else if (name === 'pincode') {
      setAddressData(prev => ({ ...prev, pincode: value }));
    }
  };
  
  const saveAddress = async () => {
    if (!selectedImage) return;
    
    setSavingAddress(true);
    setError('');
    
    try {
      // Format the address data
      const addressPayload = {
        imageId: selectedImage._id,
        name: addressData.name.trim(),
        address: {
          area: addressData.area.trim(),
          pincode: addressData.pincode.trim()
        }
      };
      
      console.log('Saving address data:', addressPayload);
      
      const result = await updateImageAddress(addressPayload);
      
      if (result.success) {
        // Update the selected image and images list with new address data
        setSelectedImage(prev => ({
          ...prev,
          name: addressData.name,
          address: {
            area: addressData.area,
            pincode: addressData.pincode
          }
        }));
        
        // Update in the images array
        setImages(prevImages => {
          return prevImages.map(img => {
            if (img._id === selectedImage._id) {
              return {
                ...img,
                name: addressData.name,
                address: {
                  area: addressData.area,
                  pincode: addressData.pincode
                }
              };
            }
            return img;
          });
        });
        
        setEditingAddress(false);
      } else {
        setError(`Failed to save address: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error saving address:', error);
      setError(`Error saving address: ${error.message || 'Unknown error'}`);
    } finally {
      setSavingAddress(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    navigate('/login');
  };

  const changePage = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  const toggleDetails = () => {
    setShowDetails(prev => !prev);
  };

  const toggleEditAddress = () => {
    setEditingAddress(prev => !prev);
  };



  // Function to handle image classification

  const handleClassify = async () => {
    if (!selectedImage) return;
    
    // Check if the image is already classified
    if (selectedImage.status === 'classified') {
      setError('This image has already been classified.');
      return;
    }
    
    setClassifying(true);
    setError('');
    setClassificationResult(null);
    setVisualizedImage(null);
    
    try {
      console.log('Classifying image:', selectedImage._id);
      
      // Call the adminClassifyImage function with multiclass mode to detect damage types
      const result = await adminClassifyImage(selectedImage._id, 'multiclass');
      
      if (result.success) {
        console.log('Classification successful:', result);
        setClassificationResult(result);
        
        // Update the selected image with the classification results
        const updatedImage = {
          ...selectedImage,
          is_road: result.classification?.is_road,
          damage_type: result.classification?.damage_type,
          severity: result.classification?.severity,
          detections: result.detections || [],
          status: 'classified'
        };
        
        setSelectedImage(updatedImage);
        
        // Also update the image in the images list
        setImages(prevImages => {
          return prevImages.map(img => 
            img._id === selectedImage._id ? updatedImage : img
          );
        });
        
        // If there's an output image with bounding boxes, display it
        if (result.output_image_url) {
          console.log('Setting visualized image from output_image_url:', result.output_image_url);
          // Create the full URL to the output image
          // Use a direct URL to the Flask backend
          const fullImageUrl = `http://localhost:5001${result.output_image_url}`;
          console.log('Full image URL:', fullImageUrl);
          setVisualizedImage(fullImageUrl);
        }
        // If no output image but detections are available, draw bounding boxes client-side
        else if (result.detections && result.detections.length > 0) {
          console.log('Drawing bounding boxes client-side');
          // Create a new image object with the updated detections
          const updatedImage = {
            ...selectedImage,
            detections: result.detections
          };
          
          // Draw the bounding boxes on the image
          drawBoundingBoxes(updatedImage);
        }
      } else {
        setError(`Classification failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error classifying image:', error);
      setError(`Error classifying image: ${error.message || 'Unknown error'}`);
    } finally {
      setClassifying(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      return dateString; // Fall back to the original string
    }
  };

  const renderClassificationResults = () => {
    if (!classificationResult) return null;

    return (
      <div className="classification-results">
        <h3>Damage Detection</h3>
        
        {/* Only show the damage detection part */}
        {classificationResult.detections && classificationResult.detections.length > 0 && (
          <div className="road-damage-results">
            <div className="damages-list">
              {classificationResult.detections.map((detection, index) => (
                <div key={index} className="damage-item">
                  <div className="damage-type">
                    <i className="fas fa-circle"></i>
                    <strong>Pothole Detected</strong>
                  </div>
                  {detection.severity && (
                    <div className={`damage-severity ${detection.severity.toLowerCase()}`}>
                      <strong>Severity:</strong> {detection.severity}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderAddressEditForm = () => {
    if (!selectedImage) return null;
    
    return (
      <div className="road-address-section">
        <div className="location-header">
          <div className="location-icon">
            <i className="fas fa-map-marker-alt"></i>
          </div>
          <h4>Road Location Information</h4>
        </div>
        
        <div className="map-visual">
          <div className="map-bg"></div>
          <div className="map-pin">
            <i className="fas fa-map-pin"></i>
          </div>
          <div className="map-circle"></div>
        </div>
        
        <div className="road-location-container">
          <div className="road-name-display">
            <div className="location-label">USER NAME</div>
            <div className="location-value">{selectedImage.name || 'Not specified'}</div>
          </div>
          
          <div className="location-details">
            <div className="area-display">
              <div className="location-label">
                <i className="fas fa-map"></i> Area/Locality
              </div>
              <div className="location-value">{getNestedValue(selectedImage, 'address.area', 'Not specified')}</div>
            </div>
            
            <div className="pincode-display">
              <div className="location-label">
                <i className="fas fa-thumbtack"></i> Pincode
              </div>
              <div className="location-value">{getNestedValue(selectedImage, 'address.pincode', 'Not specified')}</div>
            </div>
        </div>
        </div>
      </div>
    );
  };

  const renderImageDetails = () => {
    if (!selectedImage) return null;
    
    return (
      <>
        {showDetails && renderAddressEditForm()}
      </>
    );
  };

  return (
    <div className="admin-dashboard">
      <div className="dashboard-header">
        <div className="brand-section">
          <div className="logo-animation">
            <div className="logo-road"></div>
            <div className="logo-car"></div>
          </div>
          <div className="brand-text">
            <h1>ADMIN DASHBOARD</h1>
          </div>
        </div>
        <button onClick={handleLogout} className="logout-btn">
          <span className="btn-icon">⏏</span> Logout
        </button>
      </div>

      <div className="dashboard-content">
        <div className="images-section">
          <h2>User Uploaded Images</h2>
          
          {loading ? (
            <div className="loading">Loading images...</div>
          ) : error ? (
            <div className="error-message">{error}</div>
          ) : images.length === 0 ? (
            <div className="no-images">No images found in the database</div>
          ) : (
            <>
            <div className="images-grid">
                {images.map(image => (
                <div 
                  key={image._id} 
                  className={`image-card ${selectedImage && selectedImage._id === image._id ? 'selected' : ''}`}
                  onClick={() => handleImageSelect(image)}
                >
                  <div className="image-container">
                    <img 
                      src={image.image_data} 
                        alt="Uploaded road condition"
                      className="uploaded-image" 
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = 'https://via.placeholder.com/150?text=Image+Error';
                        }}
                    />
                      <div className={`status-badge ${image.status === 'classified' ? 'classified' : 'pending'}`}>
                        {image.status === 'classified' ? 'CLASSIFIED' : 'PENDING'}
                      </div>
                  </div>
                  <div className="image-info">
                      <p>{image.uploaded_by || image.email || 'Unknown user'}</p>
                      <p>Uploaded: {formatDate(image.uploaded_at || image.createdAt)}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="pagination">
              <button 
                  className="page-btn" 
                onClick={() => changePage(currentPage - 1)} 
                disabled={currentPage === 1}
              >
                Previous
              </button>
                <div className="page-info">
                  Page {currentPage} of {totalPages}
                </div>
              <button 
                  className="page-btn" 
                onClick={() => changePage(currentPage + 1)} 
                disabled={currentPage === totalPages}
              >
                Next
              </button>
            </div>
            </>
          )}
        </div>

        <div className="classification-section">
          <h2>Image Classification</h2>
          
          {!selectedImage ? (
            <div className="no-selection">
              Select an image from the left panel to classify
            </div>
          ) : (
            <div className="selected-image-container">
                <div className="selected-image-preview">
                {visualizedImage ? (
                  <img 
                    src={visualizedImage} 
                    alt="Road with damage detection" 
                    className="selected-image"
                    onError={(e) => {
                      console.error('Error loading visualized image:', visualizedImage);
                      e.target.onerror = null;
                      e.target.src = 'https://via.placeholder.com/400x300?text=Detection+Image+Error';
                    }}
                  />
                ) : (
                  <img 
                    src={`http://localhost:5001/uploads/${selectedImage.filename}`} 
                    alt="Selected road condition" 
                    className="selected-image" 
                    onError={(e) => {
                      console.error('Error loading original image');
                      e.target.onerror = null;
                      e.target.src = 'https://via.placeholder.com/400x300?text=Image+Loading+Error';
                    }}
                  />
                  )}
                </div>
              
              <div className="classification-controls">
                {selectedImage.status === 'classified' ? (
                  <div className="success-message">
                    <div className="success-icon">
                      <i className="fas fa-check-circle"></i>
                    </div>
                    <div className="success-text">
                      <h4>Success!</h4>
                      <p>Image already classified successfully.</p>
                    </div>
                  </div>
                ) : (
                  <button 
                    className="classify-btn" 
                    disabled={classifying}
                    onClick={handleClassify}
                  >
                    {classifying ? 'Classifying...' : 'Classify Image'}
                  </button>
                )}

                <button 
                  className="details-btn" 
                  onClick={toggleDetails} 
                >
                  {showDetails ? 'Hide Location Info' : 'Show Location Info'}
                </button>
              </div>
              
              {error && <div className="error-message">{error}</div>}
              
              {renderClassificationResults()}
              
              {renderImageDetails()}
            </div>
          )}
          
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
