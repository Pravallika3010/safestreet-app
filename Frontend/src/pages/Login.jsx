import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login, register } from '../services/api';
import './Login.css';

const Login = () => {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'user'
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Handle validation for register
      if (!isLogin) {
        if (formData.password !== formData.confirmPassword) {
          throw new Error('Passwords do not match');
        }
        
        if (formData.password.length < 6) {
          throw new Error('Password must be at least 6 characters');
        }
        
        // For registration, don't immediately log in
        console.log('Registering new user with email:', formData.email);
        const response = await register(formData);
        console.log('Registration response:', response);
        
        if (!response || !response.success) {
          throw new Error(response?.message || 'Registration failed. Please try again.');
        }
        
        // On successful registration, clear user data and show success message
        localStorage.removeItem('user');
        setError('');
        alert('Registration successful! Please log in with your new credentials.');
        
        // Reset form and switch to login mode
        setFormData({
          name: '',
          email: '',
          password: '',
          confirmPassword: '',
          role: 'user'
        });
        setIsLogin(true);
        setLoading(false);
        return; // Return early to prevent auto-login
      }

      // For login flow
      console.log('Attempting login with:', formData.email);
      console.log('Role:', formData.role);
      
      // Clear any existing user data before login attempt
      localStorage.removeItem('user');

      const response = await login(formData.email, formData.password, formData.role);
      console.log('Login response:', response);
      
      // Store user data in localStorage for role-based access control
      if (response && response.user) {
        // Check that we have all required fields for auth
        if (!response.user.email || !response.user.role) {
          throw new Error('Invalid user data returned from server');
        }
        
        // Make sure password is included for Basic Auth
        if (!response.user.password) {
          console.error('Password not included in user data - adding it manually');
          response.user.password = formData.password;
        }
        
        // Force save to localStorage
        try {
          const userData = {
            ...response.user,
            password: formData.password // Ensure password is included
          };
          
          localStorage.setItem('user', JSON.stringify(userData));
          console.log('Auth data stored in localStorage');
          
          // Double check storage was successful
          const storedData = localStorage.getItem('user');
          if (!storedData) {
            console.error('Failed to store user data in localStorage');
          }
        } catch (storageError) {
          console.error('Error storing auth data:', storageError);
        }
        
        // Confirm successful auth
        console.log('Authentication successful for:', response.user.email);
        
        // Small delay to ensure localStorage is updated
        setTimeout(() => {
          // Redirect based on user role
          if (response.user.role === 'admin') {
            console.log('Redirecting to admin dashboard...');
            navigate('/admin');
          } else {
            console.log('Redirecting to user dashboard...');
            navigate('/dashboard');
          }
        }, 100);
      } else {
        throw new Error('Invalid response from server - missing user data');
      }
    } catch (err) {
      console.error('Login error:', err);
      if (err.message && err.message.includes('Access denied for this role')) {
        setError('Access denied for this role');
      } else {
        setError(err.message || 'Authentication failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="home-button">
        <button onClick={() => navigate('/')} className="btn-home">
          Back to Home
        </button>
      </div>
      <div className="login-container">
        <div className="login-header">
          <h2>{isLogin ? 'SAFESTREET LOGIN' : 'Create Account'}</h2>
          <p>{isLogin ? 'Login with your credentials to access your dashboard' : 'Register for a new user account'}</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <form className="login-form" onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <div className="input-container">
                <div className="input-icon">👤</div>
                <input
                  type="text"
                  id="name"
                  name="name"
                  className="form-control"
                  placeholder="Enter your full name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
          )}
          
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <div className="input-container">
              <div className="input-icon">✉️</div>
              <input
                type="email"
                id="email"
                name="email"
                className="form-control"
                placeholder="Enter your email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="input-container">
              <div className="input-icon">🔒</div>
              <input
                type="password"
                id="password"
                name="password"
                className="form-control"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>
          </div>
          
          {!isLogin && (
            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <div className="input-container">
                <div className="input-icon">🔒</div>
                <input
                  type="password"
                  id="confirmPassword"
                  name="confirmPassword"
                  className="form-control"
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
          )}
          
          <div className="form-group">
            <label htmlFor="role">Account Type</label>
            <div className="input-container">
              <div className="input-icon">🏷️</div>
              <select
                id="role"
                name="role"
                className="form-control"
                value={formData.role}
                onChange={handleChange}
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </div>
          </div>
          
          <button
            type="submit"
            className="btn btn-primary btn-block"
            disabled={loading}
          >
            {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>
        
        <div className="divider">
          <span>or</span>
        </div>
        
        <button
          className="btn btn-outline btn-block"
          onClick={() => setIsLogin(!isLogin)}
        >
          {isLogin ? 'Create New Account' : 'Sign In Instead'}
        </button>
        
        <div className="login-footer">
          <p>
            By continuing, you agree to our Terms of Service and Privacy Policy.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
