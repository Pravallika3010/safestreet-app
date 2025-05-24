// Classify an image (admin only)
export const adminClassifyImage = async (imageId, mode = 'binary') => {
  try {
    console.log('Classifying image with ID:', imageId, 'Mode:', mode);
    
    // Create payload with the actual image ID
    const payload = { 
      image_id: imageId, 
      mode 
    };
    
    // Make the API request
    const response = await axios.post(`/api/images/admin/classify/${imageId}`, payload, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    
    console.log('Classification response:', response.data);
    
    // Return the classification results
    return response.data;
  } catch (error) {
    console.error('Classification failed:', error);
    
    // Return error information instead of mock data
    return {
      success: false,
      error: error.response?.data?.error || error.message || 'Classification failed',
      message: 'Failed to classify image. Please try again.'
    };
  }
};
