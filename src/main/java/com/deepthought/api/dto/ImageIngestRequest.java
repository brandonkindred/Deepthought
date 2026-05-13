package com.deepthought.api.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/**
 * Request body for image ingestion endpoint. Expects a base64-encoded image string.
 */
public class ImageIngestRequest {

	@Schema(description = "Base64-encoded image data", example = "/9j/4AAQSkZJRg...", required = true)
	private String image;

	public ImageIngestRequest() {
	}

	public ImageIngestRequest(String image) {
		this.image = image;
	}

	public String getImage() {
		return image;
	}

	public void setImage(String image) {
		this.image = image;
	}
}
