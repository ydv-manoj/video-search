const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SearchResult {
  video_id: string;
  video_name: string;
  text: string;
  start_time: number;
  end_time: number;
  confidence: number;
}

export interface UploadResponse {
  video_id: string;
  message: string;
  segments_processed: number;
}

export interface BatchUploadResponse {
  upload_id: string;
  message: string;
  total_videos: number;
  processing: boolean;
}

export interface UploadStatus {
  upload_id: string;
  total_videos: number;
  completed_videos: number;
  failed_videos: number;
  current_video?: string;
  status: string; // "processing", "completed", "failed", "partial"
  results: UploadResponse[];
}

export interface Video {
  video_id: string;
  video_name: string;
  segments: number;
}

class ApiService {
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  async uploadVideo(
    file: File, 
    onProgress?: (progress: number) => void,
    chunkSize: number = 8.0,
    overlap: number = 1.5
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params = new URLSearchParams({
      chunk_size: chunkSize.toString(),
      overlap: overlap.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/upload?${params}`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<UploadResponse>(response);
  }

  async uploadMultipleVideos(
    files: File[],
    chunkSize: number = 8.0,
    overlap: number = 1.5
  ): Promise<BatchUploadResponse> {
    const formData = new FormData();
    
    // Append all files
    files.forEach(file => {
      formData.append('files', file);
    });

    const params = new URLSearchParams({
      chunk_size: chunkSize.toString(),
      overlap: overlap.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/upload-batch?${params}`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<BatchUploadResponse>(response);
  }

  async getUploadStatus(uploadId: string): Promise<UploadStatus> {
    const response = await fetch(`${API_BASE_URL}/upload-status/${uploadId}`);
    return this.handleResponse<UploadStatus>(response);
  }

  async searchVideos(query: string, limit: number = 10): Promise<SearchResult[]> {
    const params = new URLSearchParams({
      query,
      limit: limit.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/search?${params}`);
    return this.handleResponse<SearchResult[]>(response);
  }

  async listVideos(): Promise<Video[]> {
    const response = await fetch(`${API_BASE_URL}/videos`);
    return this.handleResponse<Video[]>(response);
  }

  async deleteVideo(videoId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/videos/${videoId}`, {
      method: 'DELETE',
    });
    return this.handleResponse<{ message: string }>(response);
  }

  async healthCheck(): Promise<{ status: string; collection_count: number }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return this.handleResponse<{ status: string; collection_count: number }>(response);
  }

  formatTime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  generateVideoUrl(videoId: string): string {
    // In a real implementation, you'd store video files and serve them
    // For this demo, we'll use a placeholder
    return `${API_BASE_URL}/videos/${videoId}/stream`;
  }
}

export const apiService = new ApiService();