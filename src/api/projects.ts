
export const deleteProject = async (projectId: number): Promise<void> => {
  const maxRetries = 3;
  let retryCount = 0;

  while (retryCount < maxRetries) {
    try {
      const response = await axios.delete(`/api/projects/${projectId}`, {
        headers: {
          'X-User-Id': getUserId(), // Get user ID from your auth context/store
        }
      });
      
      if (response.status === 204) {
        return;
      }
      throw new Error('Failed to delete project');
    } catch (error) {
      retryCount++;
      if (retryCount === maxRetries) {
        throw new Error('Unable to connect to server. Please try again later.');
      }
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
    }
  }
};
