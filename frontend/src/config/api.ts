
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' // In production, relative path since both are deployed together
  : 'http://0.0.0.0:5000/api'; // In development, full URL

export default API_BASE_URL;
