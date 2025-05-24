import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { classifyImage, getUserImages, saveImageToDatabase, updateUserProfile, updateImageAddress } from '../services/api';
import './UserDashboard.css';

const UserDashboard = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [error, setError] = useState('');
  const [userImages, setUserImages] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [imageLoadErrors, setImageLoadErrors] = useState({});
  const [profileData, setProfileData] = useState({
    name: '',
    address: {
      area: '',
      pincode: ''
    }
  });
  const [profileMessage, setProfileMessage] = useState('');
  const [showUploads, setShowUploads] = useState(false);
  const [uploadStep, setUploadStep] = useState('upload'); // 'upload', 'address', 'complete'
  const navigate = useNavigate();

  const [classificationCache] = useState(new Map());

  const loadUserImages = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      
      console.log('Loading user images, page:', currentPage);
      
      // Check for user data in localStorage
      const userJson = localStorage.getItem('user');
      if (!userJson) {
        console.log('No user data found in localStorage, redirecting to login');
        navigate('/login');
        return;
      }
      
      // Parse user data
      try {
        const user = JSON.parse(userJson);
        console.log('Current user from localStorage:', user.email || 'unknown');
        
        if (!user || !user.email) {
          console.log('Invalid user data, redirecting to login');
          navigate('/login');
          return;
        }
      } catch (parseError) {
        console.error('Error parsing user data:', parseError);
        navigate('/login');
        return;
      }
      
      // Fetch user images from API
      try {
        const response = await getUserImages(currentPage);
        console.log('User images loaded from API:', response);
        
        if (response && response.images && Array.isArray(response.images)) {
          // Filter out any mock images or invalid data
          const validImages = response.images.filter(img => 
            img && 
            img._id && 
            img.image_data && 
            typeof img.image_data === 'string' &&
            img.image_data.length > 100 // Exclude extremely short data URLs which might be placeholders
          );
          
          console.log(`Valid images after filtering: ${validImages.length}`);
          
          // Format image data
          const formattedImages = validImages.map(img => {
            // Fix image data format if needed
            if (img.image_data && !img.image_data.startsWith('data:image')) {
              console.log(`Converting image ${img._id} to proper data URL format`);
              return { 
                ...img, 
                image_data: `data:image/jpeg;base64,${img.image_data}` 
              };
            }
            return img;
          });
          
          // Final validation to ensure all images have valid URLs
          const finalImages = formattedImages.filter(img => isValidImageUrl(img.image_data));
          
          console.log(`Found ${finalImages.length} valid images out of ${response.images.length} total`);
          setUserImages(finalImages);
          setTotalPages(response.pages || 1);
        } else {
          console.log('No images found or invalid response format');
          setUserImages([]);
          setTotalPages(1);
        }
      } catch (apiError) {
        console.error('API error loading images:', apiError);
        setError(apiError.message || 'Failed to load images from server');
        setUserImages([]);
        
        // Check if this is an authentication error
        if (apiError.message && apiError.message.includes('Authentication')) {
          localStorage.removeItem('user');
          navigate('/login');
        }
      }
    } catch (error) {
      console.error('Error in loadUserImages:', error);
      if (error.message) {
        setError(`Failed to load your images: ${error.message}`);
      } else {
        setError('Failed to load your images. Please try again.');
      }
      setUserImages([]);
    } finally {
      setLoading(false);
    }
  }, [currentPage, navigate]);

  const loadUserProfile = useCallback(() => {
    const userJson = localStorage.getItem('user');
    if (userJson) {
      try {
        const user = JSON.parse(userJson);
        setProfileData({
          name: user.name || '',
          address: user.address || { area: '', pincode: '' }
        });
      } catch (error) {
        console.error('Error parsing user data:', error);
      }
    }
  }, []);

  useEffect(() => {
    // Check if user is logged in
    const userJson = localStorage.getItem('user');
    if (!userJson) {
      console.log('No user data found in localStorage, redirecting to login');
      navigate('/login');
      return;
    }

    try {
      // Parse user data
      const user = JSON.parse(userJson);
      if (!user || !user.email) {
        console.log('Invalid user data in localStorage, redirecting to login');
        navigate('/login');
        return;
      }

      console.log('User authenticated:', user.email);
      // Load user's images
      loadUserImages();
    } catch (error) {
      console.error('Error checking authentication:', error);
      navigate('/login');
    }
  }, [navigate, loadUserImages]);

  useEffect(() => {
    loadUserProfile();
  }, [loadUserProfile]);

  // Handle image load errors
  const handleImageError = (imageId) => {
    console.error(`Failed to load image: ${imageId}`);
    setImageLoadErrors(prev => ({ ...prev, [imageId]: true }));
  };

  // Add a new function to validate image URLs
  const isValidImageUrl = (url) => {
    return url && 
           typeof url === 'string' && 
           (url.startsWith('data:image') || url.startsWith('http'));
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setClassification(null);
      setError('');
    }
  };

  // Function to resize image
  const resizeImage = (file, maxWidth = 600, maxHeight = 400) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target.result;
        
        img.onload = () => {
          // Calculate new dimensions
          let width = img.width;
          let height = img.height;
          
          if (width > maxWidth) {
            const ratio = maxWidth / width;
            width = maxWidth;
            height = height * ratio;
          }
          
          if (height > maxHeight) {
            const ratio = maxHeight / height;
            height = maxHeight;
            width = width * ratio;
          }
          
          // Create canvas and resize
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          
          // Get resized image as base64 string with reduced quality
          const resizedImage = canvas.toDataURL('image/jpeg', 0.5); // Reduced quality to 50%
          
          resolve(resizedImage);
        };
      };
    });
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    setClassification(null);
    setLoadingStep('Preparing image...');
    setLoadingProgress(10);

    try {
      console.log('Starting upload process for file:', selectedFile.name);
      console.log('File type:', selectedFile.type);
      console.log('File size:', selectedFile.size, 'bytes');
      
      // Calculate file hash for caching (using name and size as simple hash)
      const fileKey = `${selectedFile.name}-${selectedFile.size}-${selectedFile.lastModified}`;
      
      // Check if we have cached results for this image
      if (classificationCache.has(fileKey)) {
        console.log('Using cached classification result');
        setLoadingStep('Using cached result...');
        setLoadingProgress(80);
        
        const cachedResult = classificationCache.get(fileKey);
        setClassification(cachedResult);
        
        if (cachedResult.is_road) {
          setUploadStep('address');
        }
        
        setLoading(false);
        return;
      }
      
      setLoadingStep('Resizing image...');
      setLoadingProgress(20);
      console.log('Resizing image before upload...');
      const resizedImageData = await resizeImage(selectedFile);
      console.log('Image resized successfully, size:', resizedImageData.length);
      
      setLoadingStep('Classifying image...');
      setLoadingProgress(40);
      // Classify the image using the classify endpoint
      console.log('Classifying image...');
      const classificationResult = await classifyImage(resizedImageData);
      console.log('Classification result:', classificationResult);
      
      setLoadingProgress(90);
      
      // Cache the result
      const resultToCache = {
        success: classificationResult.success || false,
        is_road: classificationResult.is_road || false,
        probability: classificationResult.probability || 0,
        image_id: classificationResult.image_id
      };
      classificationCache.set(fileKey, resultToCache);
      
      if (classificationResult.success && classificationResult.is_road) {
        setLoadingStep('Classification complete!');
        setLoadingProgress(100);
        setClassification({
          success: true,
          is_road: true,
          probability: classificationResult.probability || 1.0,
          image_id: classificationResult.image_id
        });
        
        // Move to address entry step if classified as a road
        setUploadStep('address');
        setError('');
      } else {
        // If not a road, show the result but don't proceed to address step
        setLoadingStep('Not a road image');
        setLoadingProgress(100);
        setClassification({
          success: classificationResult.success || false,
          is_road: false,
          probability: classificationResult.probability || 0
        });
        setError(classificationResult.message || 'Image was not classified as a road');
      }
    } catch (err) {
      console.error('Error during upload process:', err);
      setError(err.message || 'Error processing image');
      setLoadingStep('Error');
      setLoadingProgress(0);
    } finally {
      setLoading(false);
    }
  };

  // Add a new function to save the address
  const handleSaveAddress = async () => {
    if (!classification || !classification.image_id) {
      setError('No classified image to update');
      return;
    }

    // Check if address data is available
    if (!profileData.name && (!profileData.address || (!profileData.address.area && !profileData.address.pincode))) {
      setError('Please enter road address information');
      return;
    }

    setLoading(true);
    setError('');
    setLoadingStep('Preparing address data...');
    setLoadingProgress(10);

    try {
      console.log('Saving address for image ID:', classification.image_id);
      
      // Format the road name before sending
      const formattedProfileData = {
        ...profileData,
        name: formatRoadName(profileData.name)
      };

      // Always save the complete image with address data
      // This is more reliable than trying to update an existing image
      setLoadingStep('Preparing image data...');
      setLoadingProgress(30);
      console.log('Saving full image data with address information...');
      
      // Check if we have a File or a data URL
      let imageData = previewUrl;
      
      if (selectedFile && !(typeof previewUrl === 'string' && previewUrl.startsWith('data:'))) {
        setLoadingStep('Converting image format...');
        setLoadingProgress(50);
        console.log('Converting image to data URL format...');
        // Convert the file to base64 data URL
        imageData = await resizeImage(selectedFile);
      }
      
      // Use the formatted address data with the image
      setLoadingStep('Saving image and address...');
      setLoadingProgress(70);
      const saveResult = await saveImageToDatabase(
        imageData, // The image data properly formatted
        {
          name: formattedProfileData.name,
          address: formattedProfileData.address,
          image_id: classification.image_id // Include the image_id from classification
        }
      );
      
      setLoadingProgress(90);
      console.log('Address update result:', saveResult);
      
      if (saveResult.success) {
        setLoadingStep('Save complete!');
        setLoadingProgress(100);
        setProfileMessage('');
        setUploadStep('complete');
        
        // Load updated images in the background
        loadUserImages();
      } else {
        setLoadingStep('Error saving');
        setLoadingProgress(0);
        setError(saveResult.message || 'Failed to save address information');
      }
    } catch (err) {
      console.error('Error saving address:', err);
      setError(err.message || 'Error saving address information');
      setLoadingStep('Error');
      setLoadingProgress(0);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const handleViewUploads = () => {
    setShowUploads(true);
    // Refresh images when viewing uploads
    loadUserImages();
  };

  const handleBackToUpload = () => {
    setShowUploads(false);
  };

  const changePage = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  const reloadImages = () => {
    setError('');
    loadUserImages();
  };

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    
    if (name === 'area' || name === 'pincode') {
      setProfileData(prev => ({
        ...prev,
        address: {
          ...prev.address,
          [name]: value
        }
      }));
    } else {
      setProfileData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  // Add a helper function to format road names
  const formatRoadName = (name) => {
    if (!name) return '';
    
    // Convert common abbreviations to full words
    let fullName = name
      .replace(/\bst\b\.?\s*/i, 'Street ')
      .replace(/\brd\b\.?\s*/i, 'Road ')
      .replace(/\bave\b\.?\s*/i, 'Avenue ')
      .replace(/\blvd\b\.?\s*/i, 'Boulevard ')
      .replace(/\bdr\b\.?\s*/i, 'Drive ')
      .replace(/\bln\b\.?\s*/i, 'Lane ')
      .replace(/\bct\b\.?\s*/i, 'Court ')
      .replace(/\bpkwy\b\.?\s*/i, 'Parkway ')
      .replace(/\bhwy\b\.?\s*/i, 'Highway ')
      .replace(/\bcir\b\.?\s*/i, 'Circle ')
      .replace(/\bpl\b\.?\s*/i, 'Place ')
      .trim();
    
    return fullName;
  };

  // Modify handleProfileSubmit to format road name
  const handleProfileSubmit = async () => {
    setLoading(true);
    setProfileMessage('');
    
    try {
      // Format the road name before submitting
      const formattedProfileData = {
        ...profileData,
        name: formatRoadName(profileData.name)
      };
      
      const result = await updateUserProfile(formattedProfileData);
      
      if (result.success) {
        setProfileMessage('');
        // Update local state with formatted name
        setProfileData(formattedProfileData);
      } else {
        setProfileMessage(`Failed to save address: ${result.message}`);
      }
    } catch (error) {
      setProfileMessage(`Error saving address: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Add a handleContinue function to reset the form manually
  const handleContinue = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setUploadStep('upload');
    setClassification(null);
    setProfileMessage('');
    setLoadingStep('');
    setLoadingProgress(0);
  };

  return (
    <div className="user-dashboard">
      <header className="dashboard-header">
        <div className="brand-section">
          <div className="logo-animation">
            <div className="logo-road"></div>
            <div className="logo-car"></div>
          </div>
          <div className="brand-text">
            <h1>USER DASHBOARD</h1>
          </div>
        </div>
        <div className="header-buttons">
          <button onClick={handleViewUploads} className="uploads-btn">
            <i className="btn-icon">📊</i>
            <span>My Reports</span>
          </button>
          <button onClick={handleLogout} className="logout-btn">
            <i className="btn-icon">⎋</i>
            <span>Logout</span>
          </button>
        </div>
      </header>

      {!showUploads ? (
        // Main dashboard - Upload section only
        <div className="dashboard-content-full">
          <div className="upload-section">
            <h2>Upload Road Image</h2>
            <p>Upload an image of a road for classification. Our AI will identify road surfaces for damage reporting.</p>
            
            {uploadStep === 'upload' && (
              <>
                <div className="file-input-container">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                    id="file-input"
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="file-input" className="file-input-label">
                    Choose Image
                  </label>
                  {selectedFile && (
                    <span className="file-name">{selectedFile.name}</span>
                  )}
                </div>
                
                {previewUrl && (
                  <div className="preview-container">
                    <img src={previewUrl} alt="Preview" className="preview-image" />
                  </div>
                )}

                <button 
                  onClick={handleUpload} 
                  disabled={!selectedFile || loading}
                  className="upload-button"
                >
                  {loading ? 'Processing...' : 'Analyze Image'}
                </button>
                
                {loading && (
                  <div className="loading-indicator">
                    <div className="loading-step">{loadingStep}</div>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{width: `${loadingProgress}%`}}
                      />
                    </div>
                    <div className="progress-text">{loadingProgress}%</div>
                  </div>
                )}
              </>
            )}

            {classification && (
              <div className={`classification-result ${classification.is_road ? 'success' : 'error'}`}>
                <h3>Classification Result:</h3>
                {classification.is_road ? (
                  <div className="road-classification-success">
                    <div className="classification-icon">
                      <span className="checkmark">✓</span>
                    </div>
                    <div className="classification-details">
                      <p className="classification-title">Road detected!</p>
                      {classification.probability && (
                        <div className="confidence-meter">
                          <div className="confidence-bar-container">
                            <div 
                              className="confidence-bar-fill" 
                              style={{width: `${(classification.probability * 100).toFixed(2)}%`}}
                            ></div>
                          </div>
                          <div className="confidence-percentage">{(classification.probability * 100).toFixed(2)}%</div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="road-classification-error">
                    <div className="classification-icon error-icon">
                      <span className="crossmark">❌</span>
                    </div>
                    <div className="classification-details">
                      <p className="classification-title">NOT Road</p>
                      <p>Please upload a road image.</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {uploadStep === 'address' && (
              <div className="road-address-form">
                <h3>📍 Location Details</h3>
                <p>Enhance your report with precise location information</p>
                <form className="address-form">
                  <div className="form-group">
                    <label htmlFor="name">User name: </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={profileData.name}
                      onChange={handleProfileChange}
                      className="form-control"
                      placeholder="Enter your name"
                    />
                    <span className="form-icon">👤</span>
                  </div>
                  
                  <div className="form-group">
                    <label htmlFor="area">Area/Location*</label>
                    <input
                      type="text"
                      id="area"
                      name="area"
                      value={profileData.address?.area || ''}
                      onChange={handleProfileChange}
                      className="form-control"
                      placeholder="Enter the area or locality"
                    />
                    <span className="form-icon">🏙️</span>
                  </div>
                  
                  <div className="form-group">
                    <label htmlFor="pincode">Pincode*</label>
                    <input
                      type="text"
                      id="pincode"
                      name="pincode"
                      value={profileData.address?.pincode || ''}
                      onChange={handleProfileChange}
                      className="form-control"
                      placeholder="Enter the pincode"
                    />
                    <span className="form-icon">🔢</span>
                  </div>
                  
                  <button 
                    type="button"
                    onClick={handleSaveAddress}
                    disabled={loading}
                    className="save-address-button"
                  >
                    <i>💾</i>
                    {loading ? 'Saving...' : 'Save Location Data'}
                  </button>
                  
                  {loading && (
                    <div className="loading-indicator">
                      <div className="loading-step">{loadingStep}</div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill" 
                          style={{width: `${loadingProgress}%`}}
                        />
                      </div>
                      <div className="progress-text">{loadingProgress}%</div>
                    </div>
                  )}
                </form>
              </div>
            )}


            {uploadStep === 'complete' && (
              <div className="success-message">
                <h3>Success!</h3>
                <p>Image uploaded and address information saved.</p>
                <button 
                  className="continue-button" 
                  onClick={handleContinue}
                >
                  Upload Another Image
                </button>
              </div>
            )}

            {profileMessage && (
              <div className={`profile-message ${profileMessage.includes('successfully') ? 'success' : 'error'}`}>
                {profileMessage}
              </div>
            )}
          </div>
        </div>
      ) : (
        // Uploads view
        <div className="uploads-view">
          <div className="uploads-header">
            <h2>Your Uploaded Road Images</h2>
            <div className="uploads-actions">
              <button onClick={reloadImages} className="reload-btn" disabled={loading}>
                {loading ? 'Loading...' : 'Reload Images'}
              </button>
              <button onClick={handleBackToUpload} className="back-btn">
                Back to Upload
              </button>
            </div>
          </div>
          
          {error && <div className="error-message">{error} <button onClick={reloadImages}>Try Again</button></div>}
          
          {loading && (
            <div className="loading">Loading your image gallery</div>
          )}

          {!loading && userImages.length === 0 ? (
            <div className="no-images">
              <p>No road images found in your collection. Upload new images using the form.</p>
            </div>
          ) : userImages.length > 0 ? (
            <div className="images-grid">
              {userImages.map((image, index) => (
                <div key={image._id || index} className="image-card">
                  <div className="image-container">
                    {imageLoadErrors[image._id] ? (
                      <div className="image-error">
                        <p>Image could not be loaded</p>
                        <button onClick={() => setImageLoadErrors(prev => ({ ...prev, [image._id]: false }))}>
                          Retry
                        </button>
                      </div>
                    ) : !isValidImageUrl(image.image_data) ? (
                      <div className="image-error">
                        <p>Invalid image format</p>
                      </div>
                    ) : (
                      <img 
                        src={image.image_data}
                        alt={`Road ${index + 1}`} 
                        className="uploaded-image"
                        onError={() => handleImageError(image._id)}
                      />
                    )}
                  </div>
                  <div className="image-info">
                    <p>Status: <span className={image.status || 'pending'}>{image.status || 'pending'}</span></p>
                    <p>Uploaded: {image.uploaded_at ? new Date(image.uploaded_at).toLocaleString() : 'Unknown'}</p>
                  </div>
                  {/* Add road address information at the bottom of each image card */}
                  {(image.name || (image.address && Object.values(image.address).some(val => val))) && (
                    <div className="road-address-info">
                      {image.name && <p><strong>Full Name:</strong> <span className="full-road-name">{formatRoadName(image.name)}</span></p>}
                      {image.address && image.address.area && <p><strong>Area:</strong> {image.address.area}</p>}
                      {image.address && image.address.pincode && <p><strong>Pincode:</strong> {image.address.pincode}</p>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : null}

          {totalPages > 1 && (
            <div className="pagination">
              <button 
                onClick={() => changePage(currentPage - 1)} 
                disabled={currentPage === 1}
                className="page-btn"
              >
                Previous
              </button>
              <span className="page-info">Page {currentPage} of {totalPages}</span>
              <button 
                onClick={() => changePage(currentPage + 1)} 
                disabled={currentPage === totalPages}
                className="page-btn"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UserDashboard;
