/**
 * RESTful API client for interacting with web services.
 *
 * This module provides a comprehensive HTTP client with support for
 * authentication, request/response interceptors, and error handling.
 */

const https = require('https');
const http = require('http');
const url = require('url');

/**
 * HTTP client class for making API requests
 */
class ApiClient {
    /**
     * Create a new API client instance
     * @param {string} baseURL - The base URL for all requests
     * @param {Object} options - Configuration options
     */
    constructor(baseURL, options = {}) {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': 'API-Client/1.0',
            ...options.headers
        };
        this.timeout = options.timeout || 30000;
        this.interceptors = {
            request: [],
            response: []
        };
    }

    /**
     * Add a request interceptor
     * @param {Function} interceptor - Function to modify requests
     */
    addRequestInterceptor(interceptor) {
        this.interceptors.request.push(interceptor);
    }

    /**
     * Add a response interceptor
     * @param {Function} interceptor - Function to modify responses
     */
    addResponseInterceptor(interceptor) {
        this.interceptors.response.push(interceptor);
    }

    /**
     * Make a GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async get(endpoint, options = {}) {
        return this.request('GET', endpoint, null, options);
    }

    /**
     * Make a POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async post(endpoint, data, options = {}) {
        return this.request('POST', endpoint, data, options);
    }

    /**
     * Make a PUT request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async put(endpoint, data, options = {}) {
        return this.request('PUT', endpoint, data, options);
    }

    /**
     * Make a DELETE request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async delete(endpoint, options = {}) {
        return this.request('DELETE', endpoint, null, options);
    }

    /**
     * Make an HTTP request
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    async request(method, endpoint, data, options = {}) {
        const fullURL = new URL(endpoint, this.baseURL);

        let requestOptions = {
            method,
            headers: { ...this.defaultHeaders, ...options.headers },
            body: data ? JSON.stringify(data) : undefined
        };

        // Apply request interceptors
        for (const interceptor of this.interceptors.request) {
            requestOptions = await interceptor(requestOptions);
        }

        try {
            const response = await this.makeHttpRequest(fullURL, requestOptions);

            // Apply response interceptors
            let processedResponse = response;
            for (const interceptor of this.interceptors.response) {
                processedResponse = await interceptor(processedResponse);
            }

            return processedResponse;
        } catch (error) {
            throw new ApiError(`Request failed: ${error.message}`, error.status);
        }
    }

    /**
     * Make the actual HTTP request
     * @param {URL} url - Request URL
     * @param {Object} options - Request options
     * @returns {Promise<Object>} Response data
     */
    makeHttpRequest(url, options) {
        return new Promise((resolve, reject) => {
            const protocol = url.protocol === 'https:' ? https : http;

            const request = protocol.request(url, {
                method: options.method,
                headers: options.headers,
                timeout: this.timeout
            }, (response) => {
                let data = '';

                response.on('data', (chunk) => {
                    data += chunk;
                });

                response.on('end', () => {
                    try {
                        const parsedData = JSON.parse(data);
                        resolve({
                            status: response.statusCode,
                            headers: response.headers,
                            data: parsedData
                        });
                    } catch (error) {
                        resolve({
                            status: response.statusCode,
                            headers: response.headers,
                            data: data
                        });
                    }
                });
            });

            request.on('error', (error) => {
                reject(new ApiError(`Network error: ${error.message}`));
            });

            request.on('timeout', () => {
                reject(new ApiError('Request timeout'));
            });

            if (options.body) {
                request.write(options.body);
            }

            request.end();
        });
    }

    /**
     * Set authentication token
     * @param {string} token - Authentication token
     */
    setAuthToken(token) {
        this.defaultHeaders['Authorization'] = `Bearer ${token}`;
    }

    /**
     * Remove authentication token
     */
    clearAuthToken() {
        delete this.defaultHeaders['Authorization'];
    }
}

/**
 * Custom error class for API errors
 */
class ApiError extends Error {
    constructor(message, status = null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
}

/**
 * Utility functions for API operations
 */
class ApiUtils {
    /**
     * Build query string from parameters
     * @param {Object} params - Query parameters
     * @returns {string} Query string
     */
    static buildQueryString(params) {
        const searchParams = new URLSearchParams();

        for (const [key, value] of Object.entries(params)) {
            if (value !== null && value !== undefined) {
                searchParams.append(key, String(value));
            }
        }

        return searchParams.toString();
    }

    /**
     * Parse response headers
     * @param {Object} headers - Raw headers object
     * @returns {Object} Parsed headers
     */
    static parseHeaders(headers) {
        const parsed = {};

        for (const [key, value] of Object.entries(headers)) {
            parsed[key.toLowerCase()] = value;
        }

        return parsed;
    }

    /**
     * Validate email address
     * @param {string} email - Email address to validate
     * @returns {boolean} True if valid email
     */
    static isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Generate unique request ID
     * @returns {string} Unique identifier
     */
    static generateRequestId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
}

// Export classes and functions
module.exports = {
    ApiClient,
    ApiError,
    ApiUtils
};
