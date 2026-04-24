import axios from 'axios';

/**
 * Upload image and get dog detection + emotion classification results
 * @param {File} imageFile - The image file to upload
 * @returns {Promise<Object>} Detection results
 */
export const detectEmotion = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    // IMPORTANT:
    // use relative path instead of localhost
    const response = await axios.post('/api/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;

  } catch (error) {

    if (error.response) {
      throw new Error(error.response.data.detail || 'Detection failed');

    } else if (error.request) {
      throw new Error('Cannot connect to backend API');

    } else {
      throw new Error('Unexpected error occurred');
    }
  }
};

/**
 * Check API health status
 */
export const checkHealth = async () => {

  try {
    const response = await axios.get('/health');
    return response.data;

  } catch (error) {
    throw new Error('Cannot connect to API server');
  }
};