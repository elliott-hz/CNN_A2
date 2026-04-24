import { useState, useRef } from 'react';
import { detectEmotion } from '../services/api';
import './ImageUploader.css';

const ImageUploader = ({ onResults }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      validateAndSetImage(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      validateAndSetImage(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const validateAndSetImage = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Image size must be less than 10MB');
      return;
    }

    setError(null);
    setSelectedImage(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const results = await detectEmotion(selectedImage);
      // Pass both results and image preview
      onResults(results, imagePreview);
    } catch (err) {
      setError(err.message);
      console.error('Detection error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setError(null);
    onResults(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-uploader">
      <h2>🐕 Dog Emotion Recognition</h2>
      
      {/* Upload Area */}
      <div
        className={`upload-area ${selectedImage ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        {imagePreview ? (
          <img src={imagePreview} alt="Preview" className="image-preview" />
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">📷</div>
            <p>Click or drag & drop an image here</p>
            <p className="upload-hint">Supports JPEG, PNG (max 10MB)</p>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Action Buttons */}
      {selectedImage && (
        <div className="action-buttons">
          <button 
            className="btn btn-primary" 
            onClick={handleUpload}
            disabled={isLoading}
          >
            {isLoading ? '🔍 Analyzing...' : '🚀 Detect Emotion'}
          </button>
          <button 
            className="btn btn-secondary" 
            onClick={handleReset}
            disabled={isLoading}
          >
            🔄 Reset
          </button>
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Analyzing image... This may take a few seconds on CPU</p>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
