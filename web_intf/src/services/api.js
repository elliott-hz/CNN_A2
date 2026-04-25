import axios from 'axios';

// Local development API base URL
const API_BASE_URL = 'http://localhost:8000';

/**
 * Upload image and get dog detection + emotion classification results
 * @param {File} imageFile - The image file to upload
 * @returns {Promise<Object>} Detection results
 */
export const detectEmotion = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await axios.post(`${API_BASE_URL}/api/detect`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Detection failed');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Cannot connect to server. Is the API running?');
    } else {
      // Something else happened
      throw new Error('An error occurred during detection');
    }
  }
};

/**
 * Detect emotion from base64 encoded image (optimized for video frames)
 * @param {string} base64Image - Base64 encoded image string (without data:image prefix)
 * @returns {Promise<Object>} Detection results
 */
export const detectEmotionFromBase64 = async (base64Image) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/detect-base64`, {
      image_base64: base64Image
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Detection failed');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Cannot connect to server. Is the API running?');
    } else {
      // Something else happened
      throw new Error('An error occurred during detection');
    }
  }
};

/**
 * Analyze entire video file using batch processing (FASTEST method)
 * Extracts all frames first, then processes in batches for maximum speed
 * @param {File} videoFile - The video file to analyze
 * @param {number} batchSize - Number of frames to process simultaneously (default: 10)
 * @returns {Promise<Object>} Video analysis results with all frames
 */
export const analyzeVideoBatch = async (videoFile, batchSize = 10) => {
  const formData = new FormData();
  formData.append('file', videoFile);
  
  try {
    const response = await axios.post(
      `${API_BASE_URL}/api/analyze-video-batch?batch_size=${batchSize}`, 
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout for video processing
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log(`Upload progress: ${percentCompleted}%`);
        }
      }
    );
    
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Video analysis failed');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Cannot connect to server. Is the API running?');
    } else {
      // Something else happened
      throw new Error('An error occurred during video analysis');
    }
  }
};

/**
 * Analyze entire video file and return frame-by-frame detections with real-time progress
 * @param {File} videoFile - The video file to analyze
 * @param {Function} onProgress - Callback function for progress updates (progress, currentFrame, totalFrames)
 * @returns {Promise<Object>} Video analysis results with all frames
 */
export const analyzeVideo = async (videoFile, onProgress = null) => {
  return new Promise((resolve, reject) => {
    // Create EventSource for SSE
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/api/analyze-video-stream`, true);
    
    let fullResponse = '';
    
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4) {
        if (xhr.status === 200) {
          try {
            const finalData = JSON.parse(fullResponse);
            resolve(finalData);
          } catch (e) {
            reject(new Error('Failed to parse final response'));
          }
        } else {
          reject(new Error(`Request failed with status ${xhr.status}`));
        }
      }
    };
    
    // Handle streaming data
    let lastEventId = 0;
    xhr.onprogress = function() {
      const newText = xhr.responseText.slice(lastEventId);
      const lines = newText.split('\n\n');
      
      lines.forEach(line => {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            // Check for error
            if (data.error) {
              reject(new Error(data.error));
              return;
            }
            
            // Call progress callback if provided
            if (onProgress && data.status === 'processing') {
              onProgress(data.progress, data.current_frame, data.total_frames);
            }
            
            // Store latest response
            fullResponse = line.slice(6);
            lastEventId = xhr.responseText.length - line.length;
            
          } catch (e) {
            console.error('Error parsing SSE data:', e);
          }
        }
      });
    };
    
    // Create FormData and send
    const formData = new FormData();
    formData.append('file', videoFile);
    
    xhr.send(formData);
  });
};

/**
 * Check API health status
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    throw new Error('Cannot connect to API server');
  }
};
