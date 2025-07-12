'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Search, Play, Clock, FileVideo, Trash2, Loader2, Settings, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { apiService, SearchResult, Video, BatchUploadResponse, UploadStatus } from '../../lib/api';

interface VideoUploadItem {
  file: File;
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: any;
  error?: string;
}

export default function HomePage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [chunkSize, setChunkSize] = useState(8.0);
  const [overlap, setOverlap] = useState(1.5);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Multiple upload states
  const [uploadQueue, setUploadQueue] = useState<VideoUploadItem[]>([]);
  const [batchUploadId, setBatchUploadId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<UploadStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadVideos = async () => {
    try {
      const videoList = await apiService.listVideos();
      setVideos(videoList);
    } catch (error) {
      console.error('Error loading videos:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
      const results = await apiService.searchVideos(searchQuery);
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      alert('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleFileSelection = (files: FileList | null) => {
    if (!files) return;
    
    const videoFiles = Array.from(files).filter(file => 
      file.type.startsWith('video/')
    );
    
    if (videoFiles.length === 0) {
      alert('Please select video files');
      return;
    }
    
    const newItems: VideoUploadItem[] = videoFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      status: 'pending',
      progress: 0
    }));
    
    setUploadQueue(newItems);
  };

  const startBatchUpload = async () => {
    if (uploadQueue.length === 0) return;
    
    setIsUploading(true);
    
    try {
      const files = uploadQueue.map(item => item.file);
      const response: BatchUploadResponse = await apiService.uploadMultipleVideos(files, chunkSize, overlap);
      
      setBatchUploadId(response.upload_id);
      
      // Update queue status
      setUploadQueue(prev => prev.map(item => ({
        ...item,
        status: 'processing'
      })));
      
    } catch (error) {
      console.error('Batch upload error:', error);
      alert('Upload failed. Please try again.');
      setIsUploading(false);
      setUploadQueue([]);
    }
  };

  const clearUploadQueue = () => {
    setUploadQueue([]);
    setBatchUploadId(null);
    setBatchStatus(null);
    setIsUploading(false);
  };

  const removeFromQueue = (id: string) => {
    setUploadQueue(prev => prev.filter(item => item.id !== id));
  };

  // Poll for batch upload status
  useEffect(() => {
    if (!batchUploadId || !isUploading) return;
    
    const pollStatus = async () => {
      try {
        const status = await apiService.getUploadStatus(batchUploadId);
        setBatchStatus(status);
        
        // Update individual file statuses
        setUploadQueue(prev => prev.map(item => {
          const result = status.results.find(r => r.message.includes('successfully'));
          if (result) {
            return {
              ...item,
              status: 'completed',
              progress: 100,
              result
            };
          }
          
          if (status.current_video === item.file.name) {
            return {
              ...item,
              status: 'processing',
              progress: 50
            };
          }
          
          return item;
        }));
        
        // Check if completed
        if (status.status === 'completed' || status.status === 'partial') {
          setIsUploading(false);
          loadVideos(); // Refresh video list
          
          if (status.failed_videos === 0) {
            setTimeout(() => {
              clearUploadQueue();
            }, 3000); // Auto-clear after 3 seconds if all successful
          }
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };
    
    const interval = setInterval(pollStatus, 1000); // Poll every second
    return () => clearInterval(interval);
  }, [batchUploadId, isUploading]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    handleFileSelection(e.dataTransfer.files);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const handleDeleteVideo = async (videoId: string) => {
    if (!confirm('Are you sure you want to delete this video?')) return;
    
    try {
      await apiService.deleteVideo(videoId);
      loadVideos(); // Refresh video list
      // Clear search results if they contain deleted video
      setSearchResults(prev => prev.filter(result => result.video_id !== videoId));
    } catch (error) {
      console.error('Delete error:', error);
      alert('Failed to delete video. Please try again.');
    }
  };

  const jumpToTimestamp = (videoId: string, startTime: number) => {
    setSelectedVideo(videoId);
    setCurrentTime(startTime);
  };

  // Load videos on component mount
  useEffect(() => {
    loadVideos();
  }, []);

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <Upload className="w-5 h-5 mr-2" />
            Upload Videos
          </h2>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>
        </div>

        {/* Advanced Settings */}
        {showAdvanced && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Chunking Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Chunk Size: {chunkSize}s
                </label>
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="0.5"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>3s (Precise)</span>
                  <span>15s (Context)</span>
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Overlap: {overlap}s
                </label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.5"
                  value={overlap}
                  onChange={(e) => setOverlap(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0s (No overlap)</span>
                  <span>3s (High overlap)</span>
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-600">
              <p><strong>Smaller chunks</strong> = More precise results, more segments</p>
              <p><strong>Overlap</strong> = Better context continuity across chunks</p>
              <p><strong>Gemini Embeddings</strong> = State-of-the-art semantic understanding</p>
            </div>
          </div>
        )}
        
        {/* Upload Area */}
        <div 
          className={`upload-area ${dragActive ? 'upload-area-active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            className="hidden"
            onChange={(e) => handleFileSelection(e.target.files)}
          />
          
          <div>
            <FileVideo className="w-12 h-12 mx-auto text-gray-400 mb-4" />
            <p className="text-lg text-gray-600">
              Drag and drop video files here, or click to select
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supports MP4, AVI, MOV, MKV, WebM • Multiple files supported
            </p>
            {!showAdvanced && (
              <p className="text-xs text-gray-400 mt-2">
                Will use {chunkSize}s chunks with Gemini embeddings • Click "Show Advanced" to adjust
              </p>
            )}
          </div>
        </div>

        {/* Upload Queue */}
        {uploadQueue.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">
                Upload Queue ({uploadQueue.length} files)
              </h3>
              <div className="space-x-2">
                {!isUploading && (
                  <>
                    <button
                      onClick={startBatchUpload}
                      className="btn-primary flex items-center"
                    >
                      <Upload className="w-4 h-4 mr-2" />
                      Upload All
                    </button>
                    <button
                      onClick={clearUploadQueue}
                      className="btn-secondary"
                    >
                      Clear
                    </button>
                  </>
                )}
                {isUploading && batchStatus && (
                  <div className="text-sm text-gray-600">
                    Progress: {batchStatus.completed_videos}/{batchStatus.total_videos} completed
                    {batchStatus.failed_videos > 0 && (
                      <span className="text-red-600 ml-2">
                        ({batchStatus.failed_videos} failed)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {uploadQueue.map((item) => (
                <div
                  key={item.id}
                  className={`video-upload-item ${
                    item.status === 'completed' ? 'upload-success' :
                    item.status === 'error' ? 'upload-error' :
                    item.status === 'processing' ? 'upload-processing' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {item.status === 'completed' && (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        )}
                        {item.status === 'error' && (
                          <XCircle className="w-5 h-5 text-red-600" />
                        )}
                        {item.status === 'processing' && (
                          <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                        )}
                        {item.status === 'pending' && (
                          <Clock className="w-5 h-5 text-gray-400" />
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {item.file.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {(item.file.size / (1024 * 1024)).toFixed(1)} MB
                          {item.result && (
                            <span className="ml-2 text-green-600">
                              • {item.result.segments_processed} segments processed
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                    
                    {!isUploading && item.status === 'pending' && (
                      <button
                        onClick={() => removeFromQueue(item.id)}
                        className="text-gray-400 hover:text-red-600"
                      >
                        <XCircle className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  
                  {item.status === 'processing' && (
                    <div className="progress-bar mt-2">
                      <div 
                        className="progress-bar-fill"
                        style={{ width: `${item.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Search Section */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Search className="w-5 h-5 mr-2" />
          Search Videos
        </h2>
        
        <div className="flex space-x-4">
          <input
            type="text"
            placeholder="Search for specific moments in your videos..."
            className="input-field flex-1"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button
            onClick={handleSearch}
            disabled={isSearching || !searchQuery.trim()}
            className="btn-primary flex items-center"
          >
            {isSearching ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Search className="w-4 h-4 mr-2" />
            )}
            Search
          </button>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="mt-6 space-y-4">
            <h3 className="text-lg font-medium text-gray-900">
              Search Results ({searchResults.length}) • Powered by Gemini Embeddings
            </h3>
            {searchResults.map((result, index) => (
              <div
                key={index}
                className="search-result"
                onClick={() => jumpToTimestamp(result.video_id, result.start_time)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center text-sm text-gray-500 mb-2">
                      <FileVideo className="w-4 h-4 mr-1" />
                      <span className="font-medium">{result.video_name}</span>
                      <Clock className="w-4 h-4 ml-4 mr-1" />
                      <span>
                        {apiService.formatTime(result.start_time)} - {apiService.formatTime(result.end_time)}
                      </span>
                      <span className="ml-4 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                        {Math.round(result.confidence * 100)}% match
                      </span>
                    </div>
                    <p className="text-gray-900">{result.text}</p>
                  </div>
                  <Play className="w-5 h-5 text-blue-600 ml-4 flex-shrink-0" />
                </div>
              </div>
            ))}
          </div>
        )}

        {searchResults.length === 0 && searchQuery && !isSearching && (
          <div className="mt-6 text-center text-gray-500">
            No results found for "{searchQuery}". Try different keywords.
          </div>
        )}
      </div>

      {/* Video Library */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <FileVideo className="w-5 h-5 mr-2" />
          Video Library
        </h2>
        
        {videos.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No videos uploaded yet. Upload videos to get started!
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videos.map((video) => (
              <div key={video.video_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-gray-900 truncate flex-1">
                    {video.video_name}
                  </h3>
                  <button
                    onClick={() => handleDeleteVideo(video.video_id)}
                    className="text-red-600 hover:text-red-800 ml-2"
                    title="Delete video"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-sm text-gray-500 mb-3">
                  {video.segments} searchable segments
                </p>
                <button
                  onClick={() => setSelectedVideo(video.video_id)}
                  className="w-full btn-secondary text-sm"
                >
                  <Play className="w-4 h-4 mr-1 inline" />
                  View
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Video Player Modal (Simple Implementation) */}
      {selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Video Player</h3>
              <button
                onClick={() => setSelectedVideo(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="bg-gray-900 rounded-lg p-8 text-center text-white">
              <p className="mb-4">Video Player Component</p>
              <p className="text-sm text-gray-300">
                Video ID: {selectedVideo}
              </p>
              <p className="text-sm text-gray-300">
                Current Time: {apiService.formatTime(currentTime)}
              </p>
              <p className="text-xs text-gray-400 mt-4">
                In a full implementation, this would show the actual video player
                with the ability to jump to specific timestamps.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}