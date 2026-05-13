package com.deepthought.api.dto;

import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Unit tests for ImageIngestRequest DTO.
 */
@Test(groups = "Regression")
public class ImageIngestRequestTests {

	@Test
	public void defaultConstructor_createsEmptyRequest() {
		ImageIngestRequest request = new ImageIngestRequest();
		assertNull(request.getImage());
	}

	@Test
	public void constructorWithImage_setsImage() {
		String base64 = "dGVzdA==";
		ImageIngestRequest request = new ImageIngestRequest(base64);
		assertEquals(request.getImage(), base64);
	}

	@Test
	public void setImage_updatesValue() {
		ImageIngestRequest request = new ImageIngestRequest();
		request.setImage("abc123");
		assertEquals(request.getImage(), "abc123");
		request.setImage("xyz789");
		assertEquals(request.getImage(), "xyz789");
	}

	@Test
	public void setImage_allowsNull() {
		ImageIngestRequest request = new ImageIngestRequest("data");
		request.setImage(null);
		assertNull(request.getImage());
	}
}
