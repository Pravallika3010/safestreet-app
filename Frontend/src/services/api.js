import axios from 'axios';

// Use relative URL to work with the proxy in package.json
const API_URL = '/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout (increased from 30s)
});

// Add credentials to requests if they exist
api.interceptors.request.use((config) => {
  console.log('Making request to:', config.url);
  
  // Skip auth for login and register endpoints
  if (config.url && (config.url.includes('/login') || config.url.includes('/register'))) {
    console.log('Skipping auth for login/register endpoint');
    return config;
  }
  
  // Skip auth for health and test endpoints
  if (config.url && (config.url.includes('/health') || config.url.includes('/test'))) {
    console.log('Skipping auth for health/test endpoint');
    return config;
  }
  
  // Get auth token from localStorage
  try {
    // First try to use JWT token-based auth (preferred)
    const token = localStorage.getItem('token');
    
    if (token) {
      console.log('Adding JWT token auth');
      config.headers.Authorization = `Bearer ${token}`;
      return config;
    }
    
    // Fall back to user credentials from localStorage (Basic auth)
    const userJson = localStorage.getItem('user');
    
    if (!userJson) {
      console.warn('No auth token or user data found in localStorage');
      return config;
    }
    
    let user;
    try {
      user = JSON.parse(userJson);
    } catch (parseError) {
      console.error('Error parsing user data:', parseError);
      localStorage.removeItem('user');
      return config;
    }
    
    if (user && user.email && user.password) {
      console.log('Adding Basic auth credentials for:', user.email);
      
      // Create Basic auth header - ensure proper formatting
      // This ensures correct encoding regardless of special characters
      const authString = `${user.email}:${user.password}`;
      const base64Auth = btoa(unescape(encodeURIComponent(authString)));
      config.headers.Authorization = `Basic ${base64Auth}`;
      console.log('Authorization header added');

      // Debug the auth header (without showing full password)
      const passwordHint = user.password.slice(0, 2) + '***';
      console.log(`Auth header format: Basic ${base64Auth.slice(0, 10)}... (email: ${user.email}, password: ${passwordHint})`);
    } else {
      console.warn('Incomplete user credentials found in localStorage');
      
      // Log what's missing for debugging
      console.warn('Missing:', {
        email: !user.email,
        password: !user.password
      });
    }
  } catch (error) {
    console.error('Error in auth interceptor:', error);
  }
  
  return config;
}, (error) => {
  console.error('Request interceptor error:', error);
  return Promise.reject(error);
});

// Add response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    console.log('Received response:', response.status);
    return response;
  },
  (error) => {
    console.error('API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message: error.message,
      data: error.response?.data
    });

    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.');
    }
    
    if (!error.response) {
      throw new Error('Cannot connect to server. Please check if the server is running.');
    }
    
    // Handle 401 Unauthorized errors (auth failed) but don't redirect from the interceptor
    if (error.response.status === 401) {
      // Skip for login-related endpoints
      if (error.config.url.includes('/login') || error.config.url.includes('/register')) {
        console.log('Authentication failed during login/register - not redirecting');
        throw error;
      }
      
      console.warn('Authentication error - returned 401');
      // Clear user data
      localStorage.removeItem('user');
      
      // Let the component handle the redirect
      throw new Error('Authentication failed. Please log in again.');
    }
    
    throw error;
  }
);

// Test server connection
export const testConnection = async () => {
  try {
    // First try a simple health check that doesn't require the /api prefix
    console.log('Testing connection to server health check...');
    const baseUrl = API_URL.replace('/api', '');
    const healthResponse = await axios.get(`${baseUrl}/health`);
    console.log('Health check response:', healthResponse.data);
    
    // If health check succeeds, try the regular test endpoint
    console.log('Testing connection to API endpoint:', API_URL);
    const response = await axios.get(`${API_URL}/test`);
    console.log('API test response:', response.data);
    
    return response.data;
  } catch (error) {
    console.error('Server test failed:', error);
    throw error;
  }
};

// Auth functions
export const login = async (email, password, role = 'user') => {
  try {
    console.log('Attempting login for:', email, 'with role:', role);
    
    // Clear any previous auth data
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    
    // Ensure password is a string
    const passwordStr = String(password);
    
    // Use direct login endpoint with axios to avoid auth interceptor
    console.log('Sending login request...');
    const response = await axios.post(`${API_URL}/users/login`, {
      email,
      password: passwordStr,
      role
    });
    
    console.log('Login response status:', response.status);
    console.log('Login response data:', JSON.stringify(response.data));
    
    if (!response.data) {
      throw new Error('Empty response received from server');
    }
    
    if (response.data.error) {
      throw new Error(response.data.error);
    }
    
    if (response.data && response.data.token) {
      // Store the JWT token
      localStorage.setItem('token', response.data.token);
      console.log('JWT token stored in localStorage');
    } else {
      console.warn('No token received in login response');
    }
    
    if (response.data && response.data.user) {
      // Store user data with password for Basic auth
      const userData = {
        ...response.data.user,
        password: passwordStr // Keep password for future Basic auth
      };
      
      // Validate user data before storing
      if (!userData.email) {
        throw new Error('Invalid user data received from server - missing email');
      }
      
      if (!userData._id) {
        console.warn('User data missing _id field, may cause issues with uploads');
      }
      
      // Log the exact structure we're storing
      console.log('Storing user data structure:', {
        ...userData,
        password: '***' // Hide password in logs
      });
      
      // Store in localStorage for future requests
      try {
        const userString = JSON.stringify(userData);
        localStorage.setItem('user', userString);
        console.log('User data stored in localStorage, length:', userString.length);
        
        // Verify the data was saved correctly
        const savedData = localStorage.getItem('user');
        if (!savedData) {
          console.error('Failed to retrieve user data right after saving');
        } else {
          console.log('Successfully verified user data storage');
        }
      } catch (storageError) {
        console.error('Error storing user data:', storageError);
        console.error('Storage error details:', storageError.message);
        // Continue anyway since we have the data in memory
      }
      
      return {
        ...response.data,
        success: true
      };
    } else {
      throw new Error('Invalid response format - missing user data');
    }
  } catch (error) {
    console.error('Login error:', error);
    
    // Check for specific error responses
    if (error.response) {
      console.error('Server response:', error.response.data);
      
      if (error.response.status === 404) {
        throw new Error('User not found. Please check your email or register a new account.');
      } else if (error.response.status === 401) {
        throw new Error('Invalid password. Please try again.');
      } else if (error.response.status === 403) {
        // Handle access denied / forbidden error with specific message
        throw new Error('Access denied for this role');
      } else if (error.response.data && error.response.data.error) {
        throw new Error(error.response.data.error);
      }
    }
    
    // Use the error message if it's from our own throw
    if (error.message && !error.message.includes('Network Error')) {
      throw error;
    }
    
    // Generic error
    throw new Error('Login failed. Please try again later.');
  }
};

export const register = async (userData) => {
  try {
    console.log('Attempting registration for:', userData.email);
    console.log('Registration data:', { ...userData, password: '***' }); // Log without showing password
    
    // Make sure password is a string
    if (userData.password) {
      userData.password = String(userData.password);
    }
    
    // Direct API call for registration
    const response = await axios.post(`${API_URL}/users/register`, userData);
    console.log('Registration response:', response.data);
    
    // If registration successful, return the user data but don't store in localStorage
    if (response.data && (response.data.success || response.data.message === 'User registered successfully')) {
      console.log('Registration successful for:', userData.email);
      
      return {
        success: true,
        message: 'Registration successful! Please log in with your new credentials.',
        user: response.data.user
      };
    } else {
      // Even if there's a response but not successful, return it
      console.warn('Registration response not successful:', response.data);
      return {
        success: false,
        message: response.data.error || 'Registration failed for unknown reason',
        data: response.data
      };
    }
  } catch (error) {
    console.error('Register error:', error);
    
    // Provide better error message
    if (error.response && error.response.data) {
      console.error('Error response data:', error.response.data);
      
      if (error.response.data.error) {
        throw new Error(error.response.data.error);
      }
    }
    
    throw new Error('Registration failed. Please try again.');
  }
};

export const logout = async () => {
  try {
    // No server-side logout needed with Basic auth
    // Just clear localStorage
    localStorage.removeItem('user');
    console.log('User logged out successfully');
  } catch (error) {
    console.error('Logout error:', error);
    // Make sure user data is cleared even on error
    localStorage.removeItem('user');
  }
};

export const getCurrentUser = async () => {
  try {
    const response = await api.get('/users/me');
    return response.data;
  } catch (error) {
    console.error('Get current user error:', error);
    throw error.response?.data || error;
  }
};

// Image functions
export const classifyImage = async (imageData, mode = 'binary') => {
  try {
    console.log(`Classifying image with mode: ${mode}`);
    
    const response = await api.post('/classify', {
      image: imageData,
      mode: mode
    });
    
    console.log('Classification response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Image classification error:', error);
    throw new Error(error.response?.data?.error || 'Error classifying image');
  }
};

// Add new function for road damage detection
export const detectRoadDamage = async (imageData) => {
  try {
    console.log('Detecting road damage in image');
    
    const response = await api.post('/road-damage-detection', {
      image: imageData
    });
    
    console.log('Road damage detection response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Road damage detection error:', error);
    throw new Error(error.response?.data?.error || 'Error detecting road damage');
  }
};

// Save road image to database for admin verification
export const saveImageToDatabase = async (imageData, addressData = null) => {
  try {
    console.log('Saving road image to database...');
    
    // Ensure we have image data
    if (!imageData) {
      console.error('No image data provided to saveImageToDatabase function');
      return { 
        success: false, 
        message: 'No image data provided' 
      };
    }
    
    // Log image data length and format
    console.log(`Image data length: ${imageData.length}, starts with: ${imageData.substring(0, 30)}...`);
    
    // Ensure proper data URL format
    let formattedImageData = imageData;
    if (!imageData.startsWith('data:image')) {
      console.log('Image data doesn\'t start with data:image, formatting it');
      formattedImageData = `data:image/jpeg;base64,${imageData}`;
    }
    
    // Get current user info from localStorage for logging
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    console.log(`User info for upload: email=${user.email || 'unknown'}`);
    
    // Prepare the request data with road address information if available
    const requestData = { 
      image: formattedImageData
    };
    
    // Add address data if provided
    if (addressData) {
      console.log('Address data provided:', JSON.stringify(addressData));
      
      // Ensure name is a string
      requestData.name = addressData.name ? String(addressData.name) : '';
      console.log('Name formatted:', requestData.name);
      
      // Make sure address is properly formatted as an object
      requestData.address = {
        area: addressData.address?.area ? String(addressData.address.area) : '',
        pincode: addressData.address?.pincode ? String(addressData.address.pincode) : ''
      };
      console.log('Address formatted:', JSON.stringify(requestData.address));
      
      // Add image_id if provided
      if (addressData.image_id) {
        requestData.image_id = addressData.image_id;
        console.log('Including image_id in request:', addressData.image_id);
      } else {
        console.warn('No image_id provided in addressData!');
      }
      
      console.log('Final request data with address:', JSON.stringify(requestData, null, 2));
    } else {
      console.warn('No address data provided to saveImageToDatabase!');
    }
    
    // Send the request - authentication will be added by the interceptor
    console.log('Sending save request to: /images/save');
    console.log('Request headers:', JSON.stringify(api.defaults.headers));
    
    try {
      // The api instance already has the base URL set
      // so we only need the path relative to that
      console.log('Executing POST request to /images/save...');
      const response = await api.post('/images/save', requestData);
      
      console.log('Save response received:', response.status);
      console.log('Response data:', JSON.stringify(response.data));
      
      // Check if the response contains the saved image with address
      if (response.data && response.data.image) {
        console.log('Saved image keys:', Object.keys(response.data.image));
        if (response.data.image.address) {
          console.log('Address saved successfully:', JSON.stringify(response.data.image.address));
        } else {
          console.warn('WARNING: No address in the saved image response!');
        }
      }
      
      return response.data;
    } catch (error) {
      console.error('Error during API request to /images/save:', error);
      throw error;
    }
  } catch (error) {
    console.error('Error saving image to database:', error);
    
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', JSON.stringify(error.response.data));
      return {
        success: false,
        message: error.response.data.message || 'Error saving image to database'
      };
    }
    
    return {
      success: false,
      message: error.message || 'Error communicating with the server. Please try again later.'
    };
  }
};


// Admin functions
export const adminClassifyImage = async (imageId, mode = 'multiclass') => {
  try {
    console.log('Admin classifying image:', imageId, 'with mode:', mode);
    
    // Build request headers
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    };
    
    // Create payload with the actual image ID
    const payload = { 
      image_id: imageId, 
      mode 
    };
    
    console.log('Making API request to classify image with payload:', payload);
    
    // Make the API request with explicit error handling
    try {
      const response = await axios.post(`/api/images/admin/classify/${imageId}`, payload, {
        headers: headers
      });
      
      console.log('Classification response:', response.data);
      
      // Return the classification results
      return response.data;
    } catch (apiError) {
      console.error('API request failed:', apiError);
      console.error('Response status:', apiError.response?.status);
      console.error('Response data:', apiError.response?.data);
      
      // Re-throw to be caught by outer try/catch
      throw apiError;
    }
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

// Get all images (admin only)
export const getAdminImages = async (page = 1, limit = 10) => {
  try {
    console.log('Fetching admin images, page:', page, 'limit:', limit);
    
    // Add additional debugging to track the API request
    const token = localStorage.getItem('token');
    console.log('Using auth token:', token ? 'Token available' : 'No token found');
    
    // Use the correct endpoint
    console.log('Using correct admin endpoint: /images/admin');
    const response = await api.get(`/images/admin?page=${page}&limit=${limit}`);
    
    console.log('Raw response data:', response.data);
    
    // Check if we have the expected data structure
    if (!response.data) {
      console.error('No response data received');
      return { images: [], pages: 1, total: 0 };
    }
    
    // Handle different response formats
    let processedResponse = { images: [], pages: 1, total: 0 };
    
    if (Array.isArray(response.data)) {
      // If response is an array, assume it's the images array
      console.log('Response is an array of images, converting to expected format');
      processedResponse = {
        images: response.data,
        total: response.data.length,
        pages: 1,
        page: 1,
        limit: response.data.length
      };
    } else if (response.data.images) {
      // Standard format with images property
      processedResponse = response.data;
    } else {
      // Unknown format, try to extract images
      console.warn('Unknown response format:', response.data);
      const possibleImages = Object.values(response.data).find(val => Array.isArray(val));
      if (possibleImages) {
        processedResponse.images = possibleImages;
        processedResponse.total = possibleImages.length;
      }
    }
    
    // Log the number of images received
    console.log(`Received ${processedResponse.images.length} images from API`);
    
    // Process images to ensure they have proper data URLs
    if (processedResponse.images.length > 0) {
      console.log('First image example:', {
        id: processedResponse.images[0]._id,
        hasImageData: !!processedResponse.images[0].image_data,
        status: processedResponse.images[0].status,
      });
      
      processedResponse.images = processedResponse.images.map(img => {
        // Ensure image_data has proper data URL format
        if (img.image_data && !img.image_data.startsWith('data:image')) {
          console.log(`Converting image ${img._id} to proper data URL format`);
          img.image_data = `data:image/jpeg;base64,${img.image_data}`;
        }
        return img;
      });
    }
    
    return processedResponse;
  } catch (error) {
    console.error('Error fetching admin images:', error);
    console.error('Error details:', error.response?.data || 'No response data');
    console.error('Error status:', error.response?.status || 'No status code');
    throw error.response?.data || error;
  }
};

// Get user's images
export const getUserImages = async (page = 1, limit = 10) => {
  try {
    console.log('Fetching user images, page:', page, 'limit:', limit);
    
    // Get the user info from localStorage
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    console.log('Current user for image fetch:', user.email || 'unknown');
    
    if (!user.email) {
      throw new Error('No user authenticated. Please log in again.');
    }
    
    // Use the real user images endpoint with authentication
    const response = await api.get(`/images/user?page=${page}&limit=${limit}`);
    console.log('User images response status:', response.status);
    
    // If no images are returned, provide a clean empty array
    if (!response.data || !response.data.images) {
      return { images: [], pages: 1 };
    }
    
    return response.data;
  } catch (error) {
    console.error('Error fetching user images:', error);
    
    if (error.response && error.response.status === 401) {
      throw new Error('Authentication expired. Please log in again.');
    }
    
    // Rethrow the error to be handled by the component
    throw error.response?.data || error;
  }
};

// Update image address information
export const updateImageAddress = async (addressData) => {
  try {
    console.log('Updating image address with data:', addressData);
    
    if (!addressData.imageId) {
      console.error('No image ID provided for address update');
      return {
        success: false,
        message: 'Image ID is required'
      };
    }
    
    // Log the image ID format for debugging
    console.log('Image ID format check:', {
      id: addressData.imageId,
      type: typeof addressData.imageId,
      length: addressData.imageId ? addressData.imageId.length : 0
    });
    
    // Ensure the address fields are properly formatted
    const payload = {
      image_id: addressData.imageId,
      name: addressData.name || '',
      address: {
        area: addressData.address?.area || '',
        pincode: addressData.address?.pincode || ''
      }
    };
    
    // Get user info to determine if admin
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const isAdmin = user.role === 'admin';
    
    // Use the appropriate endpoint based on user role
    const endpoint = isAdmin ? `/images/admin/update/${addressData.imageId}` : '/images/update-address';
    console.log(`Using ${isAdmin ? 'admin' : 'user'} endpoint for address update: ${endpoint}`);
    
    console.log('Sending address update request to API');
    const response = await api.put(endpoint, payload);
    
    console.log('Address update response:', response.status);
    return {
      success: true,
      message: 'Address updated successfully',
      data: response.data
    };
  } catch (error) {
    console.error('Error updating image address:', error);
    
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', JSON.stringify(error.response.data));
      return {
        success: false,
        message: error.response.data.message || 'Error updating image address'
      };
    }
    
    return {
      success: false,
      message: error.message || 'Error communicating with the server'
    };
  }
};

// Get road details by ID
export const getRoadDetails = async (roadId) => {
  try {
    console.log('Fetching road details for ID:', roadId);
    
    if (!roadId) {
      throw new Error('Road ID is required');
    }
    
    const response = await api.get(`/images/admin/image/${roadId}`);
    
    console.log('Road details response:', response.status);
    
    // If the image has image_data, ensure it has proper data URL format
    if (response.data && response.data.image_data && !response.data.image_data.startsWith('data:image')) {
      console.log('Converting image data to proper data URL format');
      response.data.image_data = `data:image/jpeg;base64,${response.data.image_data}`;
    }
    
    return response.data;
  } catch (error) {
    console.error('Error fetching road details:', error);
    
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', JSON.stringify(error.response.data));
      throw new Error(error.response.data.message || `Failed to get road details (${error.response.status})`);
    }
    
    throw new Error(error.message || 'Error communicating with the server');
  }
};

// Update user profile
export const updateUserProfile = async (profileData) => {
  try {
    console.log('Updating user profile with data:', {
      ...profileData,
      password: profileData.password ? '***' : undefined
    });
    
    const response = await api.put('/users/profile', profileData);
    
    console.log('Profile update response:', response.status);
    
    if (response.data && response.data.user) {
      // Get the stored user data first
      const storedUserJson = localStorage.getItem('user');
      if (storedUserJson) {
        const storedUser = JSON.parse(storedUserJson);
        
        // Merge the updated user with stored user data (preserving password)
        const updatedUser = {
          ...storedUser,
          ...response.data.user,
          // Keep the stored password if it exists
          password: storedUser.password
        };
        
        // Update localStorage
        localStorage.setItem('user', JSON.stringify(updatedUser));
      }
      
      return {
        success: true,
        message: 'Profile updated successfully',
        user: response.data.user
      };
    }
    
    return {
      success: false,
      message: 'Invalid response from server'
    };
  } catch (error) {
    console.error('Error updating profile:', error);
    return {
      success: false,
      message: error.response?.data?.message || 'Error updating profile'
    };
  }
};

// Test login - for debugging purposes only
export const testLogin = async () => {
  try {
    console.log('Attempting test login...');
    
    // Use direct axios call to avoid interceptors
    const response = await axios.post(`${API_URL}/test-login`);
    
    console.log('Test login response:', response.data);
    
    if (response.data && response.data.token) {
      // Store JWT token
      localStorage.setItem('token', response.data.token);
      
      // Store user data
      localStorage.setItem('user', JSON.stringify(response.data.user));
      
      console.log('Test login successful - token stored');
      return response.data;
    } else {
      throw new Error('Invalid test login response');
    }
  } catch (error) {
    console.error('Test login failed:', error);
    throw error;
  }
};

export default api; 