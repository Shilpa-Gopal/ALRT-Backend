
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-repl-name.yourusername.repl.co/api' // Update this with your actual deployed backend URL
  : 'http://0.0.0.0:5000/api';

export default API_BASE_URL;
